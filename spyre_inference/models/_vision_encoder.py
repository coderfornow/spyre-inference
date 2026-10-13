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

A model mixes ``SpyreVisionEncoderMixin`` in ahead of its upstream class and implements only
the ``vision_*`` hooks below; the protocol, warmup and serving are generic. Units: an encoder
*row* is one row of ``vision_post``'s output, and upstream's token budgets count these rows.
Host inputs have ``rows_per_token`` rows per output row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import torch.nn as nn
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import STICK, align_up
from spyre_inference.v1.worker import compile_guard

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal.encoder_budget import MultiModalBudget


@dataclass
class VisionBuffer:
    """An fp16 encoder input, ``[L]`` or ``[L, width]``, and the value of its padding rows."""

    width: int | None
    pad: float | torch.Tensor = 0.0


def offset_major(t: torch.Tensor, merge: int) -> torch.Tensor:
    """``[hp, wp, ...]`` -> ``[merge**2, tokens, ...]``: one block per offset in a merge group.

    Tokens run row-major over the merged grid, ``unfold``'s order. Rows in this layout make
    a merge or pool a free view plus a batched matmul or mean.
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
    placeholder ``__init__`` ahead of the model's in the MRO.

    The mixin owns every protocol method tied to the buffer layout (replay buffers, item
    specs), shadowing upstream's. ``select_encoder_cudagraph_items`` and
    ``get_input_modality`` come from the upstream class when it has them, else the model.
    """

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True

    # --- required: set in the model's __init__ ---
    # Host inputs by name: each is fp16 ``[L]`` or ``[L, width]`` with one row per input row,
    # and ``pad`` fills the rows a budget adds.
    vision_buffers: dict[str, VisionBuffer]
    # Input rows per encoder output row: the merge or pool factor (``M**2``), 1 if none.
    rows_per_token: int
    # Width of ``vision_post``'s output: the text model's hidden size, as the merge expects.
    out_hidden_size: int

    # --- required hooks ---

    def vision_item_rows(self, mm_kwargs: dict[str, Any]) -> list[int]:
        """Encoder output rows each item will produce, in item order.

        Implement from the item shapes in ``mm_kwargs`` (host-only, no tensor reads on the
        card). These are rows of ``vision_post``'s output before ``vision_finalize``, the unit
        of upstream's token budgets, so ``sum(rows) * rows_per_token`` input rows must equal the
        row count ``vision_inputs`` returns for the same items.
        """
        raise NotImplementedError

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Host-side preparation of every item in ``mm_kwargs`` for the encoder.

        Implement on the host: one fp16 tensor per ``vision_buffers`` key, all with the same
        row count and no padding (the mixin pads to a budget). Lay rows out in
        ``vision_row_blocks`` equal blocks (offset-major, see ``offset_major``) when
        ``vision_post`` merges or pools. Put anything data-dependent here (patching, position
        gathers, segment ids): the device graphs must depend on shapes only.
        """
        raise NotImplementedError

    def vision_pre(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Graph 1, compiled per budget: the padded device buffers -> ``(x, block_kwargs)``.

        Implement the steps before the repeated blocks (patch embedding, norms, the attention
        mask) on device tensors of static shape. ``block_kwargs`` is passed to every block
        through ``vision_call_block``.
        """
        raise NotImplementedError

    def vision_blocks(self) -> nn.ModuleList:
        """Graph 2: the repeated encoder blocks, compiled in place and shared across layers.

        Return the model's existing block list (not a copy), so weight names are unchanged.
        Swap in Spyre layers (e.g. attention) at construction, before weights load.
        """
        raise NotImplementedError

    def vision_post(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        """Graph 3, compiled per budget: the last block's output -> ``[rows, out_hidden_size]``.

        Implement the steps after the blocks (merge or pool, projector). ``taps`` holds the
        hidden states named by ``vision_taps``, in that order. Padding rows may stay in the
        output; they are dropped on the host.
        """
        raise NotImplementedError

    # --- optional: class attributes ---
    # Modalities run through the compiled graphs; any other goes to upstream's
    # embed_multimodal.
    vision_modalities: ClassVar[tuple[str, ...]] = ("image",)
    # Hidden states kept for vision_post: 0 is vision_pre's output, i the output of block i,
    # negative indices count from the last block.
    vision_taps: ClassVar[tuple[int, ...]] = ()
    # Most items one encoder call can carry, when the layout bounds it (e.g. a one-hot width).
    vision_max_items_per_call: ClassVar[int | None] = None
    # Set by the runner from vision_max_tokens_per_item before the manager is built.
    max_tokens_per_item: int

    # --- optional hooks ---

    @property
    def vision_row_blocks(self) -> int:
        """Equal blocks every buffer's rows are split into, each padded at its own tail.

        Default ``rows_per_token``: one block per merge offset, so ``vision_post`` merges with
        a free view. Override with 1 when items pad as whole units at the tail (e.g. tiles).
        """
        return self.rows_per_token

    @property
    def vision_row_quantum(self) -> int:
        """Row count every padded encoder input is a whole multiple of.

        Default: the least multiple of the 64-row stick, ``vision_row_blocks`` and
        ``rows_per_token``. Override when the layout has a larger natural unit (e.g. one
        576-row tile). Budgets are rounded up to it.
        """
        return math.lcm(STICK, self.vision_row_blocks, self.rows_per_token)

    def vision_call_block(
        self, block: nn.Module, x: torch.Tensor, block_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Run one block on ``x`` and return its output tensor.

        Default ``block(x, **block_kwargs)``. Override when the block's signature differs or it
        returns a tuple (e.g. SigLIP returns ``(hidden, None)``).
        """
        return block(x, **block_kwargs)

    def vision_finalize(
        self, out: torch.Tensor, item_rows: list[int], mm_kwargs: dict[str, Any]
    ) -> list[torch.Tensor]:
        """Host step from the encoder output to one embedding per item.

        ``out`` is ``vision_post``'s output on the host, items in order with padding rows at
        the tail; ``item_rows`` comes from ``vision_item_rows``. Default: split by
        ``item_rows``. Override when placeholders differ from encoder rows (e.g. unpadding or
        newline rows), returning tensors that match each item's placeholder count.
        """
        return list(torch.split(out[: sum(item_rows)], item_rows))

    def vision_max_tokens_per_item(self, mm_budget: MultiModalBudget) -> int:
        """Encoder output rows of the largest item the processor can produce.

        Sizes the largest warmup budget. Default: upstream's processor count
        (``mm_max_toks_per_item``), right when every placeholder is one encoder row. Override
        when they differ, from the processor's limits (e.g. the most tiles times rows per tile).
        """
        return max(mm_budget.mm_max_toks_per_item[m] for m in self.vision_modalities)

    # --- generic machinery: models do not override these ---

    def compile_vision_graphs(self) -> None:
        """Wrap ``vision_pre``, every block and ``vision_post`` in static-shape compiles and
        register them with the compile guard. Called once by the runner at warmup."""
        compile_kwargs: dict[str, Any] = dict(backend="inductor", fullgraph=True, dynamic=False)
        self._spyre_vision_pre = torch.compile(self.vision_pre, **compile_kwargs)
        self._spyre_vision_post = torch.compile(self.vision_post, **compile_kwargs)
        compile_guard.watch(self._spyre_vision_pre, f"{type(self).__name__}.vision_pre")
        compile_guard.watch(self._spyre_vision_post, f"{type(self).__name__}.vision_post")
        for block in self.vision_blocks():
            block.compile(**compile_kwargs)
            compile_guard.watch(block, f"{type(block).__name__} (vision block)")

    def _vision_empty(self, num_rows: int) -> dict[str, torch.Tensor]:
        """Host buffers of ``num_rows`` rows, filled with each buffer's padding value."""
        out = {}
        for key, buf in self.vision_buffers.items():
            shape = (num_rows,) if buf.width is None else (num_rows, buf.width)
            out[key] = torch.empty(shape, dtype=torch.float16)
            out[key][:] = buf.pad
        return out

    def _checked_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """``vision_inputs``, validated: every declared key, fp16, one row count."""
        values = self.vision_inputs(mm_kwargs)
        if set(values) != set(self.vision_buffers):
            raise ValueError(
                f"{type(self).__name__}.vision_inputs returned {sorted(values)}, but "
                f"vision_buffers declares {sorted(self.vision_buffers)}"
            )
        rows = {v.shape[0] for v in values.values()}
        if len(rows) != 1 or any(v.dtype != torch.float16 for v in values.values()):
            raise ValueError("vision_inputs must return fp16 buffers with one row count")
        return values

    # --- upstream SupportsEncoderCudaGraph, built on the hooks above ---

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        """Upstream's manager config: the modalities, buffer keys, output width, and per-block
        padding for every buffer (upstream's default would pad only at the tail)."""
        return EncoderCudaGraphConfig(
            modalities=list(self.vision_modalities),
            buffer_keys=list(self.vision_buffers),
            out_hidden_size=self.out_hidden_size,
            padding_logics={
                key: partial(_place_rows, pad=buf.pad, blocks=self.vision_row_blocks)
                for key, buf in self.vision_buffers.items()
            },
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        """Smallest and largest token budget: one row quantum, and the largest item."""
        return self.vision_row_quantum // self.rows_per_token, self.max_tokens_per_item

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        """Per-item sizes for packing into budgets, from ``vision_item_rows``. Shadows
        upstream's, whose counts follow its own buffer layout."""
        return [
            EncoderItemSpec(input_size=rows * self.rows_per_token, output_tokens=rows)
            for rows in self.vision_item_rows(mm_kwargs)
        ]

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        """The validated ``vision_inputs`` for one packed batch. Shadows upstream's, whose
        buffers would not match ``vision_buffers`` and would replay stale rows."""
        return EncoderCudaGraphReplayBuffers(values=dict(self._checked_inputs(mm_kwargs)))

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ) -> EncoderCudaGraphCaptureInputs:
        """Host buffers at one budget's row count, filled with padding values, to compile that
        budget. The manager keeps them as staging buffers; only the shape matters."""
        return EncoderCudaGraphCaptureInputs(
            values=self._vision_empty(token_budget * self.rows_per_token)
        )

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        """Run the encoder on padded device buffers: ``vision_pre``, every block, collecting
        ``vision_taps``, then ``vision_post``. Compiled after ``compile_vision_graphs``."""
        pre = getattr(self, "_spyre_vision_pre", self.vision_pre)
        post = getattr(self, "_spyre_vision_post", self.vision_post)
        blocks = self.vision_blocks()
        depth = len(blocks) + 1
        wanted = {t % depth for t in self.vision_taps}
        x, block_kwargs = pre(inputs)
        hidden = {0: x} if 0 in wanted else {}
        for i, block in enumerate(blocks, start=1):
            x = self.vision_call_block(block, x, block_kwargs)
            if i in wanted:
                hidden[i] = x
        return post(x, [hidden[t % depth] for t in self.vision_taps])

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        """The encoder without budgets, padded to one row quantum: the ``enforce_eager`` path
        and the profiling run before warmup. Not reached once compiled: the manager's startup
        check makes the largest budget cover every item."""
        values = self._checked_inputs(mm_kwargs)
        num_rows = next(iter(values.values())).shape[0]
        padded = self._vision_empty(align_up(num_rows, self.vision_row_quantum))
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
        """Copy the encoder output to the host once, then ``vision_finalize`` it into ``dest``.

        Host-side because device views at row offsets off the stick read as offset 0
        (torch-spyre#3770), and the embedding merge runs on the host anyway.
        """
        out = convert(outputs["default"], device="cpu")
        items = self.vision_finalize(
            out, [per_item_out_tokens[i] for i in indices], batch_mm_kwargs or {}
        )
        for idx, item in zip(indices, items, strict=True):
            dest[idx] = item.clone() if clone else item

    def embed_multimodal(self, **kwargs: object) -> Any:
        """Upstream's entry point when the encoder manager is not used (eager mode, profiling):
        the same hooks, uncompiled. Modalities outside ``vision_modalities`` go to upstream."""
        if self.get_input_modality(kwargs) not in self.vision_modalities:
            return super().embed_multimodal(**kwargs)
        rows = self.vision_item_rows(kwargs)
        if not rows:
            return ()
        dest: list[torch.Tensor | None] = [None] * len(rows)
        output = self.encoder_eager_forward(kwargs)
        self.postprocess_encoder_output(
            {"default": output}, list(range(len(rows))), rows, dest, batch_mm_kwargs=kwargs
        )
        return tuple(t for t in dest if t is not None)
