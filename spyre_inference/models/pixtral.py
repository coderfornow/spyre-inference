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

"""Native Mistral-format Pixtral (e.g. Ministral-3) with a compiled, bucketed vision tower.

Patch rows are laid out offset-major (see ``offset_major``). The tower is
permutation-equivariant while RoPE positions and the image mask follow the rows, so the
reorder only changes how the patch merge lowers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from vllm.model_executor.models import pixtral
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import is_vit_use_data_parallel

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    SpyreVisionEncoderMixin,
    VisionBuffer,
    offset_major,
)
from spyre_inference.multimodal.utils import (
    STICK,
    align_up,
    padded_sdpa,
    pair_swap_matrix,
    rope_rotate_matmul,
)

if TYPE_CHECKING:
    from torch import nn
    from vllm.config import VllmConfig

# Half of fp16's most negative value: a full `finfo.min` saturates to -inf and NaNs SDPA.
_MASK_MIN = torch.finfo(torch.float16).min / 2


class SpyrePixtralAttention(pixtral.Attention):
    """Upstream's attention with a matmul RoPE and the stick-aligned SDPA.

    ``freqs_cis`` is a ``(cos, sin)`` pair gathered on the host: Spyre has no complex dtype.
    """

    rope_perm: torch.Tensor

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("rope_perm", pair_swap_matrix(self.head_dim), persistent=False)

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = freqs_cis
        batch, patches, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = (
            t.reshape(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        )
        q = rope_rotate_matmul(q, cos, sin, self.rope_perm)
        k = rope_rotate_matmul(k, cos, sin, self.rope_perm)
        out = padded_sdpa(q, k, v, mask)
        out = out.transpose(1, 2).reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out


def pixtral_rope_tables(
    head_dim: int, max_side: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[max_side**2, head_dim]`` cos/sin, indexed by ``row * max_side + col``.

    The sign of the pair rotation lives in the sin table.
    """
    fc = pixtral.precompute_freqs_cis_2d(dim=head_dim, height=max_side, width=max_side, theta=theta)
    cos = fc.real.repeat_interleave(2, dim=-1)
    sin = torch.stack([-fc.imag, fc.imag], dim=-1).flatten(-2)
    return cos.reshape(-1, head_dim), sin.reshape(-1, head_dim)


def offset_major_merge_weight(weight: torch.Tensor, merge: int) -> torch.Tensor:
    """``[D_out, D * merge**2]`` with ``unfold``'s ``(c, ki, kj)`` columns ->
    ``[merge**2, D, D_out]``, one slice per offset block."""
    out_dim, in_dim = weight.shape
    return weight.reshape(out_dim, in_dim // merge**2, merge**2).permute(2, 1, 0).contiguous()


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
    SpyreVisionEncoderMixin, pixtral.PixtralForConditionalGeneration
):
    # One-hot segments: the last column is the padding rows' own image.
    vision_max_items_per_call = STICK - 1

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision(prefix)

    def _init_spyre_vision(self, prefix: str) -> None:
        args = self.vision_args
        self._merge = args.spatial_merge_size if self.patch_merger is not None else 1
        self.rows_per_token = self._merge**2
        self.out_hidden_size = self.config.text_config.hidden_size
        self._max_side = args.image_size // args.patch_size
        head_dim = args.hidden_size // args.num_attention_heads
        self._row_width = align_up(args.num_channels * args.patch_size**2)
        self.vision_buffers = {
            "rows": VisionBuffer(self._row_width),
            "segments": VisionBuffer(STICK, pad=F.one_hot(torch.tensor(STICK - 1), STICK).half()),
            "cos": VisionBuffer(head_dim),
            "sin": VisionBuffer(head_dim),
        }
        self._rope_cos, self._rope_sin = pixtral_rope_tables(
            head_dim, self._max_side, args.rope_theta
        )

        # Image limit 0 leaves a placeholder tower.
        if not isinstance(self.vision_encoder, pixtral.VisionTransformer):
            return
        for i, layer in enumerate(self.vision_encoder.transformer.layers):
            layer.attention = SpyrePixtralAttention(
                args,
                prefix=maybe_prefix(prefix, f"vision_encoder.transformer.layers.{i}.attention"),
                disable_tp=is_vit_use_data_parallel(),
            )

    def process_weights_after_loading(self) -> None:
        # Host-side, before the runner moves the model to the card.
        if not isinstance(self.vision_encoder, pixtral.VisionTransformer):
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

    # Upstream Pixtral does not implement SupportsEncoderCudaGraph, so these are ours.

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def vision_item_rows(self, mm_kwargs: dict[str, Any]) -> list[int]:
        p = self.vision_args.patch_size
        return [
            (img.shape[-2] // p) * (img.shape[-1] // p) // self.rows_per_token
            for img in _images(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        images = _images(mm_kwargs)
        return {**mm_kwargs, "images": [images[i] for i in indices]}

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Offset-major patch rows of every image, flattened as ``(c, py, px)`` like the conv
        weight, with each row's one-hot image and its gathered RoPE."""
        images = _images(mm_kwargs)
        p, m = self.vision_args.patch_size, self._merge
        rows, seg, pos = [], [], []
        for i, img in enumerate(images):
            c, h, w = img.shape
            assert h % (p * m) == 0 and w % (p * m) == 0, f"{h}x{w} is not a multiple of {p * m}"
            hp, wp = h // p, w // p
            patches = convert(img, device="cpu").reshape(c, hp, p, wp, p).permute(1, 3, 0, 2, 4)
            rows.append(offset_major(patches.reshape(hp, wp, -1), m))
            grid = torch.arange(hp)[:, None] * self._max_side + torch.arange(wp)[None, :]
            pos.append(offset_major(grid, m))
            seg.append(torch.full((m * m, (hp // m) * (wp // m)), i))
        flat = torch.cat(rows, dim=1).flatten(0, 1).to(torch.float16)
        positions = torch.cat(pos, dim=1).flatten()
        return {
            "rows": F.pad(flat, (0, self._row_width - flat.shape[1])),
            "segments": F.one_hot(torch.cat(seg, dim=1).flatten(), STICK).to(torch.float16),
            "cos": self._rope_cos[positions].to(torch.float16),
            "sin": self._rope_sin[positions].to(torch.float16),
        }

    def vision_pre(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        tower = self.vision_encoder
        patch_weight_t = cast(torch.Tensor, tower.spyre_patch_weight_t)
        x = tower.ln_pre(torch.matmul(values["rows"], patch_weight_t).unsqueeze(0))
        # A matmul, not a column-minus-row broadcast, which Spyre cannot lay out.
        segments = values["segments"]
        mask = ((1.0 - torch.matmul(segments, segments.t())) * _MASK_MIN)[None, None]
        return x, {"mask": mask, "freqs_cis": (values["cos"], values["sin"])}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_encoder.transformer.layers

    def vision_post(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        x = x.squeeze(0)
        if self.pre_mm_projector_norm is not None:
            x = self.pre_mm_projector_norm(x)
        if self.patch_merger is not None:
            x = x.reshape(self.rows_per_token, -1, x.shape[-1])
            merge_weight = cast(torch.Tensor, self.patch_merger.spyre_merge_weight)
            x = torch.matmul(x, merge_weight).sum(dim=0)
        return self.vision_language_adapter(x)
