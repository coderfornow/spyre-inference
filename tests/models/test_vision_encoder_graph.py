# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ``SpyreVisionEncoderMixin`` contract and ``SpyreEncoderGraphManager``, on CPU with a
fake model: what a model author can rely on without touching any other file."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin, VisionBuffer
from spyre_inference.v1.worker.spyre_encoder_graph import SpyreEncoderGraphManager

MERGE_PATCHES = 4  # 2x2 merge


class _Upstream(nn.Module):
    """Stands in for the upstream model class the mixin sits ahead of."""

    def get_input_modality(self, mm_kwargs):
        return mm_kwargs.get("modality", "image")

    def embed_multimodal(self, **kwargs):
        return ("upstream",)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, offset):
        return x + offset


class _FakeVisionModel(SpyreVisionEncoderMixin, _Upstream):
    """Item ``i`` is ``tokens[i]`` patches per offset group, all holding ``i + 1``."""

    def __init__(self, max_tokens: int = 300, num_blocks: int = 1) -> None:
        super().__init__()
        self.patches_per_token = MERGE_PATCHES
        self.max_tokens_per_item = max_tokens
        self.out_hidden_size = 8
        self.vision_buffers = {"patches": VisionBuffer(8), "segments": VisionBuffer(None, pad=-1.0)}
        self.blocks = nn.ModuleList([_Block() for _ in range(num_blocks)])
        self.forward_patches: list[int] = []
        self.seen_taps: list[list[torch.Tensor]] = []

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        return list(mm_kwargs["tokens"])

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        return {"tokens": [mm_kwargs["tokens"][i] for i in indices], "ids": indices}

    def vision_inputs(self, mm_kwargs):
        ids = mm_kwargs.get("ids", range(len(mm_kwargs["tokens"])))
        seg = torch.cat([torch.full((t,), float(i + 1)) for i, t in zip(ids, mm_kwargs["tokens"])])
        seg = seg.repeat(MERGE_PATCHES).to(torch.float16)
        return {"patches": seg[:, None].expand(-1, 8).contiguous(), "segments": seg}

    def vision_embed(self, values):
        return values["patches"], {"offset": 0.0}

    def vision_blocks(self):
        return self.blocks

    def vision_project(self, x, taps):
        self.forward_patches.append(x.shape[0])
        self.seen_taps.append(taps)
        return x.reshape(MERGE_PATCHES, -1, x.shape[-1]).mean(dim=0)


def _vllm_config(budgets=None, tp_mode="weights"):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=budgets or [],
            encoder_cudagraph_max_vision_items_per_batch=0,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda modality: 0 if modality == "video" else 4,
                mm_encoder_tp_mode=tp_mode,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )


def _manager(model, **kwargs) -> SpyreEncoderGraphManager:
    return SpyreEncoderGraphManager(
        _vllm_config(**kwargs), torch.device("cpu"), torch.float16, model
    )


def test_budgets_are_stick_aligned_and_cover_the_largest_item():
    manager = _manager(_FakeVisionModel(max_tokens=300))
    assert manager.token_budgets == [16, 32, 64, 128, 256, 304]
    assert all(b * MERGE_PATCHES % 64 == 0 for b in manager.token_budgets)


def test_user_budgets_below_the_largest_item_fail_at_startup():
    with pytest.raises(ValueError, match="largest encoder budget"):
        _manager(_FakeVisionModel(max_tokens=300), budgets=[16, 64])


def test_encoder_data_parallel_is_rejected():
    with pytest.raises(NotImplementedError, match="mm_encoder_tp_mode"):
        _manager(_FakeVisionModel(), tp_mode="data")


def test_execute_packs_pads_per_block_and_splits_per_item():
    model = _FakeVisionModel(max_tokens=300)
    manager = _manager(model)
    manager.capture(graph_pool=None)
    assert sorted(model.forward_patches) == sorted(b * MERGE_PATCHES for b in manager.token_budgets)

    model.forward_patches.clear()
    outputs = manager.execute({"tokens": [10, 30]})

    # 40 tokens pack into the 64-token budget: one compiled call at 256 patches.
    assert model.forward_patches == [64 * MERGE_PATCHES]
    assert [o.shape[0] for o in outputs] == [10, 30]
    for i, out in enumerate(outputs):
        assert out.device.type == "cpu"
        torch.testing.assert_close(out, torch.full_like(out, float(i + 1)))


def test_unrecorded_budget_is_a_miss():
    manager = _manager(_FakeVisionModel())
    assert manager._run_budget_graph({"tokens": [10]}, 16) is None


def test_inputs_must_match_the_declared_buffers():
    """An upstream ``prepare_encoder_cudagraph_replay_buffers`` with other keys would
    otherwise replay stale padding silently."""
    model = _FakeVisionModel()
    model.vision_inputs = lambda mm_kwargs: {"patches": torch.zeros(4, 8, dtype=torch.float16)}
    with pytest.raises(ValueError, match="vision_buffers declares"):
        model.prepare_encoder_cudagraph_replay_buffers({"tokens": [1]}, 0, 0)


def test_inputs_must_be_fp16_with_one_patch_count():
    model = _FakeVisionModel()
    model.vision_inputs = lambda mm_kwargs: {
        "patches": torch.zeros(4, 8),
        "segments": torch.zeros(4, dtype=torch.float16),
    }
    with pytest.raises(ValueError, match="fp16"):
        model.prepare_encoder_cudagraph_replay_buffers({"tokens": [1]}, 0, 0)


def test_items_per_call_cap_splits_the_batch():
    model = _FakeVisionModel()
    type(model).vision_max_items_per_call = 1
    try:
        manager = _manager(model)
        manager.capture(graph_pool=None)
        model.forward_patches.clear()
        outputs = manager.execute({"tokens": [10, 30]})
    finally:
        del type(model).vision_max_items_per_call
    assert manager.max_batch_size == 1
    assert len(model.forward_patches) == 2
    assert [o.shape[0] for o in outputs] == [10, 30]


def test_taps_reach_vision_project_in_declared_order():
    model = _FakeVisionModel(num_blocks=3)
    type(model).vision_taps = (-1, 0, 2)
    try:
        bufs = model._vision_empty(64)
        bufs["patches"][:] = 1.0
        model.blocks[0].forward = lambda x, offset: x + 1
        model.blocks[1].forward = lambda x, offset: x + 10
        model.blocks[2].forward = lambda x, offset: x + 100
        model.encoder_cudagraph_forward(bufs)
    finally:
        del type(model).vision_taps
    (taps,) = model.seen_taps
    assert [float(t[0, 0]) for t in taps] == [112.0, 1.0, 12.0]


def test_postprocess_override_gets_host_output_and_shapes_each_item():
    """A model reshapes its embeddings through upstream's ``postprocess_encoder_output``."""
    model = _FakeVisionModel()
    seen = []

    def postprocess(outputs, indices, tokens, dest, clone=False, batch_mm_kwargs=None):
        out = outputs["default"]
        seen.append(out.device.type)
        offset = 0
        for i in indices:
            dest[i] = out[offset + 1 : offset + tokens[i]]
            offset += tokens[i]

    model.postprocess_encoder_output = postprocess
    embeds = model.embed_multimodal(tokens=[10, 30])
    assert seen == ["cpu"]
    assert [e.shape[0] for e in embeds] == [9, 29]


def test_budget_range_override_sets_the_largest_budget():
    """A model whose processor count is not in encoder tokens overrides upstream's hook."""
    model = _FakeVisionModel(max_tokens=300)
    model.get_encoder_cudagraph_budget_range = lambda vllm_config: (16, 500)
    manager = _manager(model)
    assert manager.token_budgets[-1] >= 500


def test_other_modalities_go_to_upstream():
    assert _FakeVisionModel().embed_multimodal(modality="audio") == ("upstream",)
