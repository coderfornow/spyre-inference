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

"""Compiled, bucketed vision encoders on upstream's ``SupportsEncoderCudaGraph`` protocol.

A model mixes ``SpyreVisionEncoderMixin`` in ahead of its upstream class and supplies the
protocol's host-side methods (item specs, item selection, replay buffers) plus three device
graphs: ``vision_pre``, ``vision_blocks`` and ``vision_post``. ``SpyreEncoderGraphManager``
compiles them once per upstream token budget at warmup.
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
    """An fp16 encoder input, ``[L]`` or ``[L, width]``, and the value of its padding rows."""

    width: int | None
    pad: float | torch.Tensor = 0.0


def offset_major(t: torch.Tensor, merge: int) -> torch.Tensor:
    """``[hp, wp, ...]`` -> ``[merge**2, tokens, ...]``: one block per offset in a merge group.

    Tokens run row-major over the merged grid, ``unfold``'s order. Rows in this layout make
    a patch merge a free view plus a batched matmul.
    """
    hp, wp = t.shape[:2]
    t = t.reshape(hp // merge, merge, wp // merge, merge, *t.shape[2:])
    t = t.permute(1, 3, 0, 2, *range(4, t.dim()))
    return t.reshape(merge * merge, (hp // merge) * (wp // merge), *t.shape[4:])


def _place_rows(
    dst: torch.Tensor, src: torch.Tensor, pad: float | torch.Tensor, blocks: int
) -> None:
    """Copy ``src`` into the larger ``dst`` block by block, padding each block's tail."""
    d = dst.view(blocks, -1, *dst.shape[1:])
    s = src.view(blocks, -1, *src.shape[1:])
    d[:] = pad
    d[:, : s.shape[1]].copy_(s)


class SpyreVisionEncoderMixin:
    """Structural ``SupportsEncoderCudaGraph``: subclassing the protocol would put its
    placeholder ``__init__`` ahead of the model's in the MRO. Methods an upstream model
    already implements (item specs, item selection, modality) are left to it."""

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True
    vision_buffers: dict[str, VisionBuffer]
    rows_per_token: int
    # Set by the runner from upstream's processor budget before the manager is built.
    max_tokens_per_item: int
    out_hidden_size: int

    def vision_pre(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Device inputs -> ``(x, block_kwargs)``."""
        raise NotImplementedError

    def vision_blocks(self) -> nn.ModuleList:
        raise NotImplementedError

    def vision_post(self, x: torch.Tensor) -> torch.Tensor:
        """Block output -> ``[L / rows_per_token, out_hidden_size]``."""
        raise NotImplementedError

    @property
    def row_align(self) -> int:
        return math.lcm(STICK, self.rows_per_token)

    def compile_vision_graphs(self) -> None:
        compile_kwargs: dict[str, Any] = dict(backend="inductor", fullgraph=True, dynamic=False)
        self._spyre_vision_pre = torch.compile(self.vision_pre, **compile_kwargs)
        self._spyre_vision_post = torch.compile(self.vision_post, **compile_kwargs)
        compile_guard.watch(self._spyre_vision_pre, f"{type(self).__name__}.vision_pre")
        compile_guard.watch(self._spyre_vision_post, f"{type(self).__name__}.vision_post")
        for block in self.vision_blocks():
            block.compile(**compile_kwargs)
            compile_guard.watch(block, f"{type(block).__name__} (vision block)")

    def _vision_empty(self, num_rows: int) -> dict[str, torch.Tensor]:
        out = {}
        for key, buf in self.vision_buffers.items():
            shape = (num_rows,) if buf.width is None else (num_rows, buf.width)
            out[key] = torch.empty(shape, dtype=torch.float16)
            out[key][:] = buf.pad
        return out

    # --- upstream SupportsEncoderCudaGraph ---

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=list(self.vision_buffers),
            out_hidden_size=self.out_hidden_size,
            # Upstream's default pads at the tail; these rows pad per offset block.
            padding_logics={
                key: partial(_place_rows, pad=buf.pad, blocks=self.rows_per_token)
                for key, buf in self.vision_buffers.items()
            },
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
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
        # Host tensors, kept by the manager as staging buffers; only the shape matters.
        return EncoderCudaGraphCaptureInputs(
            values=self._vision_empty(token_budget * self.rows_per_token)
        )

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        pre = getattr(self, "_spyre_vision_pre", self.vision_pre)
        post = getattr(self, "_spyre_vision_post", self.vision_post)
        x, block_kwargs = pre(inputs)
        for block in self.vision_blocks():
            x = block(x, **block_kwargs)
        return post(x)

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        values = self.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
        padded = self._vision_empty(align_up(next(iter(values.values())).shape[0], self.row_align))
        padding = self.get_encoder_cudagraph_config().padding_logics
        for key, src in values.items():
            padding[key](padded[key], src)
        device = next(self.vision_blocks().parameters()).device
        return self.encoder_cudagraph_forward(
            {k: convert(v, device=device) for k, v in padded.items()}
        )

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Slice on the host: device views at row offsets off the stick read as offset 0
        # (torch-spyre#3770), and the embedding merge runs on the host anyway.
        out = convert(outputs["default"], device="cpu")
        scatter_output_slices(out, indices, per_item_out_tokens, dest, clone)

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        tokens = [spec.output_tokens for spec in self.get_encoder_cudagraph_item_specs(kwargs)]
        if not tokens:
            return ()
        dest: list[torch.Tensor | None] = [None] * len(tokens)
        output = self.encoder_eager_forward(kwargs)
        self.postprocess_encoder_output({"default": output}, list(range(len(tokens))), tokens, dest)
        return tuple(t for t in dest if t is not None)
