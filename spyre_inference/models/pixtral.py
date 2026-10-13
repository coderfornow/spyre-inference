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

"""Native Mistral-format Pixtral (e.g. Ministral-3) with a bucketed, compiled vision tower."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from vllm.model_executor.models.pixtral import (
    PixtralForConditionalGeneration,
    VisionTransformer,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from spyre_inference.models.vision.base import SpyreVisionEncoderMixin, VisionBuffer
from spyre_inference.models.vision.pixtral import (
    SpyrePixtralAttention,
    offset_major_merge_weight,
    pixtral_patch_rows,
    pixtral_rope_tables,
    segment_pad_row,
)
from spyre_inference.multimodal.utils import STICK, align_up

if TYPE_CHECKING:
    from torch import nn
    from vllm.config import VllmConfig

# Half of fp16's most negative value: a full `finfo.min` saturates to -inf and NaNs SDPA.
_MASK_MIN = torch.finfo(torch.float16).min / 2


def _linear_weight(layer: nn.Module) -> torch.Tensor:
    """``[out, in]``, undoing the Spyre linear's post-load transpose and row padding."""
    weight = cast(torch.Tensor, layer.weight).detach()
    padding = getattr(layer, "spyre_row_padding", None)
    if padding is None:
        return weight
    weight = weight.t()
    return weight[: weight.shape[0] - padding]


def _images(mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
    images = mm_kwargs.get("images")
    if images is None:
        return []
    return list(images.unbind(0)) if isinstance(images, torch.Tensor) else list(images)


class SpyrePixtralForConditionalGeneration(
    SpyreVisionEncoderMixin, PixtralForConditionalGeneration
):
    """Upstream Pixtral with Spyre attention and the three-graph vision encoder.

    Processor, weight loading, placeholders and ``embed_input_ids`` are upstream's.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision(prefix)

    def _init_spyre_vision(self, prefix: str) -> None:
        args = self.vision_args
        self._merge = args.spatial_merge_size if self.patch_merger is not None else 1
        self.rows_per_token = self._merge**2
        self.out_hidden_size = self.config.text_config.hidden_size
        self._max_side = args.image_size // args.patch_size
        self.max_tokens_per_item = (self._max_side // self._merge) ** 2
        head_dim = args.hidden_size // args.num_attention_heads
        self._row_width = align_up(args.num_channels * args.patch_size**2)
        self.vision_buffers = {
            "rows": VisionBuffer(self._row_width),
            # Padding rows get their own segment, so they never join an image's block.
            "segments": VisionBuffer(STICK, pad=segment_pad_row(STICK)),
            "cos": VisionBuffer(head_dim),
            "sin": VisionBuffer(head_dim),
        }
        # Host-only: the gather by position runs in input preparation.
        self._rope_cos, self._rope_sin = pixtral_rope_tables(
            head_dim, self._max_side, args.rope_theta
        )

        # Text-only runs (image limit 0) get a placeholder tower.
        if not isinstance(self.vision_encoder, VisionTransformer):
            return
        for i, layer in enumerate(self.vision_encoder.transformer.layers):
            layer.attention = SpyrePixtralAttention(
                args,
                prefix=maybe_prefix(prefix, f"vision_encoder.transformer.layers.{i}.attention"),
                disable_tp=is_vit_use_data_parallel(),
            )

    def process_weights_after_loading(self) -> None:
        # Runs on the host, before the runner moves the model to the card.
        if not isinstance(self.vision_encoder, VisionTransformer):
            return
        tower = self.vision_encoder
        weight = tower.patch_conv.weight.detach().flatten(1)
        weight_t = F.pad(weight, (0, self._row_width - weight.shape[1])).t().contiguous()
        tower.register_buffer("spyre_patch_weight_t", weight_t, persistent=False)
        if self.patch_merger is not None:
            merge_weight = offset_major_merge_weight(
                _linear_weight(self.patch_merger.merging_layer), self._merge
            )
            self.patch_merger.register_buffer("spyre_merge_weight", merge_weight, persistent=False)

    # --- upstream SupportsEncoderCudaGraph, model part (host) ---

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        p = self.vision_args.patch_size
        specs = []
        for img in _images(mm_kwargs):
            rows = (img.shape[-2] // p) * (img.shape[-1] // p)
            specs.append(
                EncoderItemSpec(input_size=rows, output_tokens=rows // self.rows_per_token)
            )
        return specs

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        images = _images(mm_kwargs)
        return {**mm_kwargs, "images": [images[i] for i in indices]}

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        values = pixtral_patch_rows(
            _images(mm_kwargs),
            patch_size=self.vision_args.patch_size,
            merge=self._merge,
            max_side=self._max_side,
            row_width=self._row_width,
            segment_width=STICK,
            cos_table=self._rope_cos,
            sin_table=self._rope_sin,
        )
        return EncoderCudaGraphReplayBuffers(values=dict(values))

    # --- the three device graphs ---

    def vision_pre(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        tower = self.vision_encoder
        patch_weight_t = cast(torch.Tensor, tower.spyre_patch_weight_t)
        x = tower.ln_pre(torch.matmul(values["rows"], patch_weight_t).unsqueeze(0))
        segments = values["segments"]
        same_image = torch.matmul(segments, segments.t())
        mask = ((1.0 - same_image) * _MASK_MIN)[None, None]
        return x, {"mask": mask, "freqs_cis": (values["cos"], values["sin"])}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_encoder.transformer.layers

    def vision_post(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(0)
        if self.pre_mm_projector_norm is not None:
            x = self.pre_mm_projector_norm(x)
        if self.patch_merger is not None:
            # Offset-major rows: one [tokens, D] block per merge offset, each with its slice.
            x = x.reshape(self.rows_per_token, -1, x.shape[-1])
            merge_weight = cast(torch.Tensor, self.patch_merger.spyre_merge_weight)
            x = torch.matmul(x, merge_weight).sum(dim=0)
        return self.vision_language_adapter(x)
