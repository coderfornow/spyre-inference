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


def build_rope_perm(kind: str, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """CPU `[dim, dim]` permutation `M` so `x @ M` is rope's rotation shuffle.

    A matmul rather than a slice: `cat([x[..., half:], x[..., :half]], -1)` returns
    uncorrelated data on device whenever `x` came from a matmul, and at head_dim=64 a
    `d/2`-wide half cannot be laid out at all ("Unexpected stick expression ...
    Mod(var, 32)").
    """
    m = torch.zeros(dim, dim, dtype=dtype)
    if kind == "pair":
        even = torch.arange(0, dim, 2)
        m[even, even + 1] = 1.0
        m[even + 1, even] = 1.0
    elif kind == "half_swap":
        half = dim // 2
        rows = torch.cat([torch.arange(half, dim), torch.arange(0, half)])
        m[rows, torch.arange(dim)] = 1.0
    else:
        raise ValueError(f"unknown rope permutation kind {kind!r}")
    return m


_ROPE_PERM_OP_CACHE: dict[tuple, torch.Tensor] = {}


@torch.library.custom_op("spyre_inference::rope_perm_matrix", mutates_args=())
def rope_perm_op(kind: str, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """`build_rope_perm` as one opaque node, for a permutation that cannot be warmed.

    Every tower today warms its permutation before the first trace and reads it from a
    dict, which costs nothing; this is the escape hatch for a constant whose shape is
    only known inside a compiled region. It costs one fallback node per call site per
    forward, so it does not belong on a path a warm call can reach.

    No tensor argument, on purpose: torch-spyre's coarse-tile scheduler may only
    relocate a tensor-input-free fallback (`_is_tensor_input_free_fallback` in
    `torch_spyre/_inductor/wsr/coarse_tile_hints.py`), and a tensor operand would pin
    the node inside the tiled walk.
    """
    key = (kind, dim, dtype, str(device))
    m = _ROPE_PERM_OP_CACHE.get(key)
    if m is None:
        m = convert(build_rope_perm(kind, dim, dtype), device=device, dtype=dtype)
        _ROPE_PERM_OP_CACHE[key] = m
    return m


@rope_perm_op.register_fake
def _rope_perm_op_fake(
    kind: str, dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.empty(dim, dim, dtype=dtype, device=device)


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
