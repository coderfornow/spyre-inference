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

"""Native (Mistral-format) Pixtral vision tower on Spyre.

Rows are laid out on the host offset-major: for each offset ``g = ki*M + kj`` inside an
``M x M`` merge group, one contiguous block of every token's patch at that offset. The
patch merger is then a free view to ``[M**2, tokens, D]``, a batched matmul and a sum,
where upstream's ``unfold`` regroup cannot lower. The tower is permutation-equivariant
as long as RoPE positions and the image mask follow the rows, so the reorder changes
nothing numerically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.models import pixtral

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import (
    padded_sdpa,
    rope_perm_matrix,
    rope_rotate_matmul,
)


class SpyrePixtralAttention(pixtral.Attention):
    """Pixtral vision attention with a matmul RoPE and the stick-aligned SDPA.

    ``freqs_cis`` is a ``(cos, sin)`` pair of ``[L, head_dim]`` tensors gathered on the
    host, instead of upstream's complex table, which Spyre cannot represent.
    """

    rope_perm: torch.Tensor

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("rope_perm", rope_perm_matrix("pair", self.head_dim), persistent=False)

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = freqs_cis
        batch, patches, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        q = rope_rotate_matmul(q, cos, sin, self.rope_perm)
        k = rope_rotate_matmul(k, cos, sin, self.rope_perm)
        out = padded_sdpa(q, k, v, mask)
        out = out.transpose(1, 2).reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out


def pixtral_rope_tables(
    head_dim: int, max_side: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` tables ``[max_side**2, head_dim]`` for ``rope_rotate_matmul``.

    Indexed by ``row * max_side + col``. The sin table carries the pair-rotation sign,
    so ``x*cos + pair_swap(x)*sin`` equals upstream's complex multiply.
    """
    fc = pixtral.precompute_freqs_cis_2d(dim=head_dim, height=max_side, width=max_side, theta=theta)
    cos = fc.real.repeat_interleave(2, dim=-1)
    sin = torch.stack([-fc.imag, fc.imag], dim=-1).flatten(-2)
    return cos.reshape(-1, head_dim), sin.reshape(-1, head_dim)


def offset_major(t: torch.Tensor, merge: int) -> torch.Tensor:
    """``[hp, wp, ...]`` -> ``[M**2, tokens, ...]``: offset ``(ki, kj)``, then merged token.

    Tokens run row-major over the merged ``(hp/M, wp/M)`` grid, ``unfold``'s order.
    """
    hp, wp = t.shape[:2]
    rest = t.shape[2:]
    t = t.reshape(hp // merge, merge, wp // merge, merge, *rest)
    t = t.permute(1, 3, 0, 2, *range(4, t.dim()))
    return t.reshape(merge * merge, (hp // merge) * (wp // merge), *rest)


def segment_pad_row(width: int) -> torch.Tensor:
    """Padding rows' one-hot: the last column is reserved for them."""
    row = torch.zeros(width, dtype=torch.float16)
    row[-1] = 1.0
    return row


def pixtral_patch_rows(
    images: list[torch.Tensor],
    *,
    patch_size: int,
    merge: int,
    max_side: int,
    row_width: int,
    segment_width: int,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Host S1-in: patch rows, one-hot image segments and gathered RoPE, unpadded in L.

    Rows are offset-major across all images (``offset_major``, concatenated per offset).
    Each patch flattens as ``(c, py, px)``, matching ``patch_conv.weight.flatten(1)``;
    the row width is zero-padded to ``row_width``. ``segments[r, i] = 1`` when row ``r``
    belongs to image ``i``, so ``segments @ segments.T`` is the block-diagonal mask; a
    matmul because Spyre cannot lay out a column-minus-row broadcast.
    """
    assert len(images) < segment_width, (
        f"{len(images)} images in one encoder call; at most {segment_width - 1} fit"
    )
    rows, seg, pos = [], [], []
    for i, img in enumerate(images):
        img = convert(img, device="cpu")
        c, h, w = img.shape
        assert h % (patch_size * merge) == 0 and w % (patch_size * merge) == 0, (
            f"image {h}x{w} is not a multiple of patch_size*merge={patch_size * merge}"
        )
        hp, wp = h // patch_size, w // patch_size
        patches = img.reshape(c, hp, patch_size, wp, patch_size).permute(1, 3, 0, 2, 4)
        rows.append(offset_major(patches.reshape(hp, wp, -1), merge))
        grid = torch.arange(hp)[:, None] * max_side + torch.arange(wp)[None, :]
        pos.append(offset_major(grid, merge))
        seg.append(torch.full((merge * merge, (hp // merge) * (wp // merge)), i))
    flat = torch.cat(rows, dim=1).flatten(0, 1).to(torch.float16)
    positions = torch.cat(pos, dim=1).flatten()
    return {
        "rows": F.pad(flat, (0, row_width - flat.shape[1])),
        "segments": F.one_hot(torch.cat(seg, dim=1).flatten(), segment_width).to(torch.float16),
        "cos": cos_table[positions].to(torch.float16),
        "sin": sin_table[positions].to(torch.float16),
    }


def offset_major_merge_weight(weight: torch.Tensor, merge: int) -> torch.Tensor:
    """Merging-layer weight ``[D_out, D*M**2]`` (``unfold``'s ``(c, ki, kj)`` columns) as
    ``[M**2, D, D_out]``, one ``D -> D_out`` slice per offset."""
    out_dim, in_dim = weight.shape
    return weight.reshape(out_dim, in_dim // merge**2, merge**2).permute(2, 1, 0).contiguous()
