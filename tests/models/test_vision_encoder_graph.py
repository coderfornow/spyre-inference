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

"""``SpyreEncoderGraphManager`` on upstream's budget logic, with a fake model on CPU.

Upstream's packing and padding run unchanged; these pin the Spyre parts: stick-aligned
budgets, the startup coverage check, per-block padding and the compiled-graph replay.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers, EncoderItemSpec

from spyre_inference.models.vision.base import SpyreVisionEncoderMixin, VisionBuffer
from spyre_inference.v1.worker.spyre_encoder_graph import SpyreEncoderGraphManager

MERGE_ROWS = 4  # 2x2 merge


class _FakeVisionModel(SpyreVisionEncoderMixin, nn.Module):
    """Each item is ``tokens`` merged tokens whose rows hold the item's index + 1."""

    def __init__(self, max_tokens: int = 300) -> None:
        nn.Module.__init__(self)
        self.rows_per_token = MERGE_ROWS
        self.max_tokens_per_item = max_tokens
        self.out_hidden_size = 8
        self.vision_buffers = {"rows": VisionBuffer(8), "segments": VisionBuffer(None, pad=-1.0)}
        self.blocks = nn.ModuleList([nn.Identity()])
        self.forward_rows: list[int] = []

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        return [
            EncoderItemSpec(input_size=t * MERGE_ROWS, output_tokens=t) for t in mm_kwargs["tokens"]
        ]

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        return {"tokens": [mm_kwargs["tokens"][i] for i in indices], "ids": indices}

    def prepare_encoder_cudagraph_replay_buffers(
        self, mm_kwargs, max_batch_size, max_frames, path="default"
    ):
        ids = mm_kwargs.get("ids", range(len(mm_kwargs["tokens"])))
        # Offset-major: per offset block, every item's tokens in order.
        seg = torch.cat([torch.full((t,), float(i + 1)) for i, t in zip(ids, mm_kwargs["tokens"])])
        seg = seg.repeat(MERGE_ROWS)
        rows = seg[:, None].expand(-1, 8).to(torch.float16)
        return EncoderCudaGraphReplayBuffers(
            values={"rows": rows, "segments": seg.to(torch.float16)}
        )

    def vision_pre(self, values):
        return values["rows"], {}

    def vision_blocks(self):
        return self.blocks

    def vision_post(self, x):
        self.forward_rows.append(x.shape[0])
        return x.reshape(MERGE_ROWS, -1, x.shape[-1]).mean(dim=0)


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


def test_budgets_are_stick_aligned_and_cover_the_largest_image():
    manager = _manager(_FakeVisionModel(max_tokens=300))
    assert manager.token_budgets == [16, 32, 64, 128, 256, 304]
    assert all(b * MERGE_ROWS % 64 == 0 for b in manager.token_budgets)


def test_user_budgets_below_the_largest_image_fail_at_startup():
    with pytest.raises(ValueError, match="largest encoder budget"):
        _manager(_FakeVisionModel(max_tokens=300), budgets=[16, 64])


def test_encoder_data_parallel_is_rejected():
    with pytest.raises(NotImplementedError, match="mm_encoder_tp_mode"):
        _manager(_FakeVisionModel(), tp_mode="data")


def test_execute_packs_pads_per_block_and_splits_per_item():
    model = _FakeVisionModel(max_tokens=300)
    manager = _manager(model)
    manager.capture()
    assert sorted(model.forward_rows) == sorted(b * MERGE_ROWS for b in manager.token_budgets)

    model.forward_rows.clear()
    outputs = manager.execute({"tokens": [10, 30]})

    # 40 tokens pack into the 64-token budget: one compiled call at 256 rows.
    assert model.forward_rows == [64 * MERGE_ROWS]
    assert [o.shape[0] for o in outputs] == [10, 30]
    # Per-block padding keeps each item's rows on its own tokens in every block.
    for i, out in enumerate(outputs):
        assert out.device.type == "cpu"
        torch.testing.assert_close(out, torch.full_like(out, float(i + 1)))


def test_unrecorded_budget_is_a_miss():
    model = _FakeVisionModel()
    manager = _manager(model)
    assert manager._run_budget_graph({"tokens": [10]}, 16) is None
