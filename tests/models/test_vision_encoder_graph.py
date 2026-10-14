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

import dataclasses
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


def _vllm_config(budgets=None, tp_mode="weights", videos=0):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=budgets or [],
            encoder_cudagraph_max_vision_items_per_batch=0,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda modality: videos if modality == "video" else 4,
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
    with pytest.raises(ValueError, match="declared dtypes"):
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


def test_items_per_call_cap_also_caps_the_frames():
    """Upstream sizes the frame limit from the uncapped item count before the cap applies."""
    model = _FakeVisionModel()
    config = model.get_encoder_cudagraph_config()
    model.get_encoder_cudagraph_config = lambda: dataclasses.replace(config, max_frames_per_video=8)
    type(model).vision_max_items_per_call = 2
    try:
        manager = _manager(model, videos=1)
    finally:
        del type(model).vision_max_items_per_call
    assert manager.max_batch_size == 2
    assert manager.max_frames_per_batch == 2 * 8


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


def test_integer_buffers_keep_their_dtype_and_padding():
    """Index buffers (e.g. positions) are padded and validated in their declared dtype."""
    model = _FakeVisionModel()
    model.vision_buffers = {
        **model.vision_buffers,
        "positions": VisionBuffer(None, pad=0, dtype=torch.int64),
    }
    bufs = model._vision_empty(64)
    assert bufs["positions"].dtype == torch.int64

    inputs = model.vision_inputs

    def with_positions(mm_kwargs):
        values = inputs(mm_kwargs)
        return {**values, "positions": torch.arange(values["patches"].shape[0])}

    model.vision_inputs = with_positions
    values = model.prepare_encoder_cudagraph_replay_buffers({"tokens": [10]}, 0, 0).values
    model.get_encoder_cudagraph_config().padding_logics["positions"](
        bufs["positions"], values["positions"]
    )
    assert bufs["positions"].view(MERGE_PATCHES, -1)[:, 10:].eq(0).all()


def test_gather_table_gathers_rows_on_the_host():
    from spyre_inference.models._vision_encoder import GatherTable

    table = GatherTable(torch.arange(12.0).reshape(4, 3))
    torch.testing.assert_close(
        table.gather(torch.tensor([2, 0])), torch.tensor([[6.0, 7.0, 8.0], [0.0, 1.0, 2.0]])
    )
    assert table.half().table.dtype == torch.float16


def test_out_of_range_tap_fails_instead_of_aliasing():
    """A tap past the last layer would otherwise wrap onto an earlier one by modulo."""
    model = _FakeVisionModel(num_blocks=2)
    type(model).vision_taps = (3,)
    try:
        with pytest.raises(ValueError, match="out of range for 3 hidden states"):
            model.encoder_cudagraph_forward(model._vision_empty(64))
    finally:
        del type(model).vision_taps


def test_compile_vision_graphs_twice_keeps_the_first_compile():
    """A retried warmup must not re-wrap the graphs it already compiled."""
    model = _FakeVisionModel(num_blocks=2)
    model.compile_vision_graphs()
    embed, project = model._spyre_vision_embed, model._spyre_vision_project
    blocks = [b._compiled_call_impl for b in model.blocks]
    model.compile_vision_graphs()
    assert model._spyre_vision_embed is embed
    assert model._spyre_vision_project is project
    assert [b._compiled_call_impl for b in model.blocks] == blocks


def test_item_and_frame_caps_together_through_capture_and_execute():
    """Both caps reach capture, and replay still splits the batch at the item cap."""
    model = _FakeVisionModel()
    config = model.get_encoder_cudagraph_config()
    model.get_encoder_cudagraph_config = lambda: dataclasses.replace(config, max_frames_per_video=8)
    capture_limits = []
    prepare = model.prepare_encoder_cudagraph_capture_inputs

    def recording_prepare(budget, max_batch_size, max_frames_per_batch, *args, **kwargs):
        capture_limits.append((max_batch_size, max_frames_per_batch))
        return prepare(budget, max_batch_size, max_frames_per_batch, *args, **kwargs)

    model.prepare_encoder_cudagraph_capture_inputs = recording_prepare
    type(model).vision_max_items_per_call = 2
    try:
        manager = _manager(model, videos=1)
        manager.capture(graph_pool=None)
        model.forward_patches.clear()
        outputs = manager.execute({"tokens": [5, 10, 15, 20, 25]})
    finally:
        del type(model).vision_max_items_per_call
    assert set(capture_limits) == {(2, 16)}
    assert len(model.forward_patches) == 3
    assert [o.shape[0] for o in outputs] == [5, 10, 15, 20, 25]
    for i, out in enumerate(outputs):
        torch.testing.assert_close(out, torch.full_like(out, float(i + 1)))


def test_each_path_compiles_and_replays_from_its_own_staging():
    """A model declaring two upstream paths gets one staging set per path and budget."""
    from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphPathConfig, EncoderItemSpec

    model = _FakeVisionModel()
    config = model.get_encoder_cudagraph_config()
    paths = {"local": EncoderCudaGraphPathConfig(), "global": EncoderCudaGraphPathConfig()}
    model.get_encoder_cudagraph_config = lambda: dataclasses.replace(config, paths=paths)
    model.get_encoder_cudagraph_item_specs = lambda mm_kwargs: [
        EncoderItemSpec(t * MERGE_PATCHES, t, {"local": t, "global": t})
        for t in mm_kwargs["tokens"]
    ]
    forward = model.encoder_cudagraph_forward
    seen_paths = []

    def recording_forward(inputs, path="default"):
        seen_paths.append(path)
        return forward(inputs) + (100.0 if path == "global" else 0.0)

    model.encoder_cudagraph_forward = recording_forward
    model.postprocess_encoder_output = lambda outputs, indices, tokens, dest, **kw: [
        dest.__setitem__(i, outputs["local"][:t] + outputs["global"][:t])
        for i, t in zip(indices, [tokens[i] for i in indices])
    ]

    manager = _manager(model)
    manager.capture(graph_pool=None)
    assert set(manager._staging) == {"local", "global"}
    assert all(set(s) == set(manager.token_budgets) for s in manager._staging.values())

    seen_paths.clear()
    (out,) = manager.execute({"tokens": [10]})
    assert sorted(seen_paths) == ["global", "local"]
    torch.testing.assert_close(out, torch.full_like(out, 1.0 + 101.0))


def test_runner_warmup_builds_the_manager_and_serves_through_it():
    """The runner path: _record_vision_graphs compiles the model, builds the Spyre manager
    from the processor's largest item and captures every budget."""
    from vllm.config import CompilationMode

    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    model = _FakeVisionModel()
    compiled = []
    model.compile_vision_graphs = lambda: compiled.append(True)
    vllm_config = _vllm_config()
    vllm_config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    vllm_config.compilation_config.inductor_compile_config = {}
    vllm_config.model_config.enforce_eager = False
    vllm_config.model_config.dtype = torch.float16

    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.vllm_config = vllm_config
    runner.compilation_config = vllm_config.compilation_config
    runner.model_config = vllm_config.model_config
    runner.supports_mm_inputs = True
    runner.model = SimpleNamespace(_model=model)
    runner.mm_budget = SimpleNamespace(mm_max_toks_per_item={"image": 300})
    runner._spyre_device = torch.device("cpu")
    runner.encoder_cudagraph_manager = None

    runner._record_vision_graphs()

    manager = runner.encoder_cudagraph_manager
    assert compiled == [True]
    assert isinstance(manager, SpyreEncoderGraphManager)
    assert model.max_tokens_per_item == 300
    assert manager.token_budgets[-1] == 304
    assert set(manager._staging["default"]) == set(manager.token_budgets)
    outputs = manager.execute({"tokens": [10, 30]})
    assert [o.shape[0] for o in outputs] == [10, 30]
    assert manager.graph_hits == 2 and manager.graph_misses == 0
