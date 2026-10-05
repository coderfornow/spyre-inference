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

"""Helpers shared by the vision-tower workarounds in this package."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.utils import convert

# Spyre stick width in 2-byte elements (128-byte stick). Matmul reduction dims and
# the sequence axis must land on it.
STICK = 64


def align_up(n: int, align: int = STICK) -> int:
    return (n + align - 1) // align * align


# Attribute under which a source mask caches its padded counterpart `(key, padded)`.
_MASK_ATTR = "_spyre_padded_mask"


def padded_attn_mask(
    mask: torch.Tensor,
    b: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """The additive `[b, 1, seq_pad, seq_pad]` mask `padded_sdpa` expects, on `device`.

    Callers build it once per image, outside their block loop: a host-resident tensor
    reaching a compiled block has no device layout for torch-spyre to lower.

    O(L²) and shared by every layer, so it is cached on the source mask: one upload
    per image, released with its source.
    """
    seq_pad = align_up(seq)
    key = (b, seq, dtype, str(device))
    cached = getattr(mask, _MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    # Assembled on CPU: strided slice-assign is not stick-safe on Spyre.
    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
    m[:, :, :, seq:] = neg_inf  # padded keys never attended
    mc = convert(mask, "cpu")
    if mc.dtype == torch.bool:
        m[:, :, :seq, :seq] = torch.zeros(seq, seq, dtype=dtype).masked_fill(
            ~mc.reshape(seq, seq), neg_inf
        )
    else:
        m[:, :, :seq, :seq] = mc.to(dtype).reshape(seq, seq)

    m = convert(m, device)
    setattr(mask, _MASK_ATTR, (key, m))
    return m


def padded_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """SDPA over `[B, H, L, D]` with L and D padded to the stick, then cropped.

    At a sequence length coprime with the stick, stock SDPA either fails to
    restickify a batch-matmul operand or returns silently wrong values, so the
    padding is a correctness requirement rather than a tuning choice. Padded keys are
    masked to `-inf` and padded queries cropped off.

    `mask` is the already-padded additive mask from `padded_attn_mask`, or `None` to
    attend everywhere — then only the padded key columns need masking, which a
    `[1, 1, 1, seq_pad]` row built on device does without any O(L²) mask.

    `scale` defaults to the head dim seen here, which assumes `q`/`k`/`v` arrive unpadded
    so the padding cannot change it. Pass it explicitly when the head dim is already
    padded, or when the model carries its own scale.
    """
    seq, d = q.shape[-2:]
    if scale is None:
        scale = d**-0.5
    seq_pad = align_up(seq)
    d_pad = align_up(d)
    padded = (seq_pad, d_pad) != (seq, d)

    if mask is None and seq_pad != seq:
        neg_inf = torch.finfo(q.dtype).min
        mask = F.pad(q.new_zeros(1, 1, 1, seq), (0, seq_pad - seq), value=neg_inf)

    if padded:
        # F.pad's tuple runs from the last dim backwards: (D left, D right, L left, L right).
        pad = (0, d_pad - d, 0, seq_pad - seq)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
    else:
        # Offset operands read as offset 0 (torch-spyre#3770), so SDPA is silently
        # wrong here; the padded branch escapes it only because F.pad materializes.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    if padded:
        # Offset-0 prefix slice, so torch-spyre#3770 cannot bite. Left as a view: the
        # caller's transpose+reshape materializes it anyway.
        out = out[:, :, :seq, :d]
    return out
