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

"""Bucketed, compiled vision encoders on upstream's ``SupportsEncoderCudaGraph`` protocol.

Upstream's ``EncoderCudaGraphManager`` already picks token budgets, packs images into
them and pads replay buffers; only its capture and replay are CUDA-specific. On Spyre a
budget is the static shape the encoder is compiled at (``SpyreEncoderGraphManager``).

A model mixes this in ahead of its upstream class and supplies:

- upstream's ``get_encoder_cudagraph_item_specs``, ``select_encoder_cudagraph_items``
  and ``prepare_encoder_cudagraph_replay_buffers`` (host-side input preparation);
- ``vision_buffers``, ``rows_per_token``, ``max_tokens_per_item``, ``out_hidden_size``.
  Every buffer is indexed by row, and its rows are ``rows_per_token`` equal blocks
  (one per offset inside a merge group), each padded at its own tail, so a merge is
  a view in the post graph;
- ``vision_pre`` / ``vision_blocks`` / ``vision_post``: the three device graphs. Split
  so all blocks share one compiled graph per bucket, where CUDA captures one graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import torch.nn as nn
from vllm.model_executor.models.utils import scatter_output_slices
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
)

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import STICK, align_up
from spyre_inference.v1.worker import compile_guard

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class VisionBuffer:
    """One fp16 encoder input indexed by row: ``[L]`` if ``width`` is None, else ``[L, width]``."""

    width: int | None
    pad: float | torch.Tensor = 0.0
    """Value of every padding row: a scalar, or one ``[width]`` row."""


def _place_rows(
    dst: torch.Tensor, src: torch.Tensor, pad: float | torch.Tensor, blocks: int
) -> None:
    """Copy ``src`` into the larger ``dst`` block by block, padding each block's tail."""
    d = dst.view(blocks, -1, *dst.shape[1:])
    s = src.view(blocks, -1, *src.shape[1:])
    d[:] = pad
    d[:, : s.shape[1]].copy_(s)


class SpyreVisionEncoderMixin:
    """Implements upstream's ``SupportsEncoderCudaGraph`` structurally (it is a
    runtime-checkable protocol). Not a subclass: protocols carry a placeholder
    ``__init__`` that would sit ahead of the model's in the MRO and skip it."""

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True
    vision_buffers: dict[str, VisionBuffer]
    rows_per_token: int
    max_tokens_per_item: int
    out_hidden_size: int

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def vision_pre(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Graph 1: device inputs -> ``(x, block_kwargs)`` for every block."""
        raise NotImplementedError

    def vision_blocks(self) -> nn.ModuleList:
        """Graph 2: the repeated blocks, called as ``block(x, **block_kwargs)``."""
        raise NotImplementedError

    def vision_post(self, x: torch.Tensor) -> torch.Tensor:
        """Graph 3: block output -> ``[L / rows_per_token, out_hidden_size]``."""
        raise NotImplementedError

    @property
    def row_align(self) -> int:
        """Row multiple every L is padded to: whole sticks and whole tokens."""
        return math.lcm(STICK, self.rows_per_token)

    def compile_vision_graphs(self) -> None:
        compile_kwargs: dict[str, Any] = dict(backend="inductor", fullgraph=True, dynamic=False)
        pre = torch.compile(self.vision_pre, **compile_kwargs)
        post = torch.compile(self.vision_post, **compile_kwargs)
        compile_guard.watch(pre, f"{type(self).__name__}.vision_pre")
        compile_guard.watch(post, f"{type(self).__name__}.vision_post")
        for block in self.vision_blocks():
            block.compile(**compile_kwargs)
            compile_guard.watch(block, f"{type(block).__name__} (vision block)")
        self._spyre_vision_pre = pre
        self._spyre_vision_post = post

    def _vision_forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        pre = getattr(self, "_spyre_vision_pre", self.vision_pre)
        post = getattr(self, "_spyre_vision_post", self.vision_post)
        x, block_kwargs = pre(values)
        for block in self.vision_blocks():
            x = block(x, **block_kwargs)
        return post(x)

    def _vision_device(self) -> torch.device:
        return next(self.vision_blocks().parameters()).device

    def _vision_empty(self, num_rows: int) -> dict[str, torch.Tensor]:
        out = {}
        for key, buf in self.vision_buffers.items():
            shape = (num_rows,) if buf.width is None else (num_rows, buf.width)
            out[key] = torch.empty(shape, dtype=torch.float16)
            out[key][:] = buf.pad
        return out

    # --- upstream SupportsEncoderCudaGraph, generic part ---

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=list(self.vision_buffers),
            out_hidden_size=self.out_hidden_size,
            # Every key: upstream's default appends padding at the tail, not per block.
            padding_logics={
                key: partial(_place_rows, pad=buf.pad, blocks=self.rows_per_token)
                for key, buf in self.vision_buffers.items()
            },
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        # The floor is one stick of rows: no bucket below it can be laid out anyway.
        return self.row_align // self.rows_per_token, self.max_tokens_per_item

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ) -> EncoderCudaGraphCaptureInputs:
        # Host tensors: the Spyre manager keeps them as staging buffers and uploads.
        # Compiled shapes depend only on L, so the padding values are enough.
        return EncoderCudaGraphCaptureInputs(
            values=self._vision_empty(token_budget * self.rows_per_token)
        )

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        return self._vision_forward(inputs)

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        """Pad only to ``row_align`` and run uncompiled: the ``enforce_eager`` path."""
        values = self.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
        num_rows = next(iter(values.values())).shape[0]
        padded = self._vision_empty(align_up(num_rows, self.row_align))
        config = self.get_encoder_cudagraph_config()
        device = self._vision_device()
        for key, src in values.items():
            config.padding_logics[key](padded[key], src)
        return self._vision_forward({k: convert(v, device=device) for k, v in padded.items()})

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # One D2H, then slice on the host: device views at row offsets off the stick
        # read as offset 0 (torch-spyre#3770), and the embedding merge runs on the host.
        out = convert(outputs["default"], device="cpu")
        scatter_output_slices(out, indices, per_item_out_tokens, dest, clone)

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        specs = self.get_encoder_cudagraph_item_specs(kwargs)
        if not specs:
            return ()
        tokens = [spec.output_tokens for spec in specs]
        dest: list[torch.Tensor | None] = [None] * len(specs)
        self.postprocess_encoder_output(
            {"default": self.encoder_eager_forward(kwargs)}, list(range(len(specs))), tokens, dest
        )
        return tuple(t for t in dest if t is not None)
