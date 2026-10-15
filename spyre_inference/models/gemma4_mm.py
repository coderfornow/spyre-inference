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

"""Gemma 4 with its vision tower compiled per token budget.

``SpyreGemma4ForConditionalGeneration`` keeps upstream's processor, weight loading and
decoder, and adds the ``SpyreVisionEncoderMixin`` hooks:

- attention: ``SpyreGemma4VisionAttention``, swapped in by class so vLLM's linear
  replacement and the weight names are untouched; the head dim is padded to two sticks,
  RoPE is a matmul against a half-swap matrix, SDPA is the stick-aligned one;
- norms: ``SpyreGemma4RMSNorm``, in the input dtype;
- inputs: the processor's row-major patches laid out offset-major per image, with each
  patch's image id and (x, y) position;
- embedding: the patch linear as a matmul, the position table and the RoPE tables gathered
  on the card by position;
- projection: the pooler's average as a mean over the offset groups, the standardize
  affine, then ``embed_vision``.

Images only. ``models.gemma4.default_mm_limits`` sets the video limit to 0 before launch,
so a video never reaches the tower.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma4 import modeling_gemma4
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    GatherTable,
    SpyreVisionEncoderMixin,
    VisionBuffer,
    offset_major,
)
from spyre_inference.multimodal.utils import (
    STICK,
    align_up,
    padded_sdpa,
    rope_perm_matrix,
    rope_rotate_matmul,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Half of fp16's most negative value: a full `finfo.min` saturates to -inf and NaNs SDPA.
_MASK_MIN = torch.finfo(torch.float16).min / 2


# --- weight padding: the head dim to two sticks, the MLP width to one ---


def _host(t: torch.Tensor) -> torch.Tensor:
    """Composed strided slice-assignments lower silently wrong on a Spyre-resident tensor."""
    return convert(t.detach(), device="cpu")


def _as_plain_linear(layer: nn.Module) -> nn.Linear:
    """vLLM's `recursive_replace_linear` makes `Gemma4ClippableLinear.linear` a Spyre
    replicated linear whose weight is stored transposed; the padding helpers assume
    nn.Linear's `[out, in]`. A quantized layer fails rather than being dequantized."""
    if isinstance(layer, nn.Linear):
        return layer
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    quant_method = getattr(layer, "quant_method", None)
    if quant_method is not None and not isinstance(quant_method, UnquantizedLinearMethod):
        raise NotImplementedError(
            "Gemma 4 vision head-dim padding cannot preserve the quantization method "
            f"on {type(layer).__name__} ({type(quant_method).__name__}); it rebuilds "
            "the projection as a plain nn.Linear. Run the vision tower unquantized."
        )
    weight_t = _host(cast(torch.Tensor, layer.weight))
    in_features, out_features = weight_t.shape
    plain = nn.Linear(in_features, out_features, bias=layer.bias is not None)
    plain.weight = nn.Parameter(weight_t.t().contiguous(), requires_grad=False)
    if layer.bias is not None:
        plain.bias = nn.Parameter(
            _host(cast(torch.Tensor, layer.bias)).clone(), requires_grad=False
        )
    return plain


def _linear_weight(layer: nn.Module) -> torch.Tensor:
    """`[out, in]` on the host, undoing the Spyre linear's post-load transpose and row padding."""
    weight = _host(cast(torch.Tensor, layer.weight))
    padding = getattr(layer, "spyre_row_padding", None)
    if padding is None:
        return weight
    weight = weight.t()
    return weight[: weight.shape[0] - padding]


def _assert_clipping_preserves_zero(proj, name: str) -> None:
    """Padded lanes must stay exactly zero: `_padded_rms_norm` rescales the variance on
    that basis and the zero columns of `o_proj`/`down_proj` ignore them."""
    if not getattr(proj, "use_clipped_linears", False):
        return
    lo = _host(proj.output_min)
    hi = _host(proj.output_max)
    if not bool(torch.all((lo <= 0) & (hi >= 0)).item()):
        raise NotImplementedError(
            f"{name}: output clipping must include zero when the head dim is padded, "
            f"or padded channels stop being zero before RMSNorm. Got "
            f"output_min={lo.tolist()}, output_max={hi.tolist()}."
        )


def _pad_qk_linear(proj, num_heads: int, orig_head_dim: int, padded_head_dim: int) -> nn.Linear:
    """Each head's `[X, Y]` quarters packed as `[X, Y, 0 | X, Y, 0]`, so one half swap
    rotates both position axes at once."""
    linear = proj.linear
    weight = _host(linear.weight).view(num_heads, orig_head_dim, -1)
    new_weight = torch.zeros(num_heads, padded_head_dim, weight.shape[-1], dtype=weight.dtype)
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    new_weight[:, :quarter] = weight[:, :quarter]
    new_weight[:, quarter : 2 * quarter] = weight[:, 2 * quarter : 3 * quarter]
    new_weight[:, padded_half : padded_half + quarter] = weight[:, quarter : 2 * quarter]
    new_weight[:, padded_half + quarter : padded_half + 2 * quarter] = weight[:, 3 * quarter :]
    padded = nn.Linear(
        linear.in_features, num_heads * padded_head_dim, bias=linear.bias is not None
    )
    padded.weight = nn.Parameter(
        new_weight.reshape(num_heads * padded_head_dim, -1), requires_grad=False
    )
    if linear.bias is not None:
        bias = _host(linear.bias).view(num_heads, orig_head_dim)
        new_bias = torch.zeros(num_heads, padded_head_dim, dtype=bias.dtype)
        new_bias[:, :quarter] = bias[:, :quarter]
        new_bias[:, quarter : 2 * quarter] = bias[:, 2 * quarter : 3 * quarter]
        new_bias[:, padded_half : padded_half + quarter] = bias[:, quarter : 2 * quarter]
        new_bias[:, padded_half + quarter : padded_half + 2 * quarter] = bias[:, 3 * quarter :]
        padded.bias = nn.Parameter(new_bias.reshape(-1), requires_grad=False)
    return padded


def _pad_proj_output_simple(
    proj: nn.Linear, n_heads: int, orig_head_dim: int, padded_head_dim: int
) -> nn.Linear:
    w = _host(proj.weight)
    hidden = w.shape[1]
    new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
    for h in range(n_heads):
        s, d = h * orig_head_dim, h * padded_head_dim
        new_w[d : d + orig_head_dim, :] = w[s : s + orig_head_dim, :]
    new_proj = nn.Linear(hidden, n_heads * padded_head_dim, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        bias = _host(proj.bias)
        new_b = torch.zeros(n_heads * padded_head_dim, dtype=bias.dtype)
        for h in range(n_heads):
            s, d = h * orig_head_dim, h * padded_head_dim
            new_b[d : d + orig_head_dim] = bias[s : s + orig_head_dim]
        new_proj.bias = nn.Parameter(new_b, requires_grad=False)
    return new_proj


def _pad_proj_input_simple(
    proj: nn.Linear, n_heads: int, orig_head_dim: int, padded_head_dim: int
) -> nn.Linear:
    w = _host(proj.weight)
    hidden = w.shape[0]
    new_w = torch.zeros(hidden, n_heads * padded_head_dim, dtype=w.dtype)
    for h in range(n_heads):
        s, d = h * orig_head_dim, h * padded_head_dim
        new_w[:, d : d + orig_head_dim] = w[:, s : s + orig_head_dim]
    new_proj = nn.Linear(n_heads * padded_head_dim, hidden, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        new_proj.bias = nn.Parameter(_host(proj.bias).clone(), requires_grad=False)
    return new_proj


def _pad_norm_weight(norm, orig_head_dim: int, padded_head_dim: int) -> nn.Parameter:
    weight = _host(norm.weight)
    padded = torch.ones(padded_head_dim, dtype=weight.dtype)
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    padded[:quarter] = weight[:quarter]
    padded[quarter : 2 * quarter] = weight[2 * quarter : 3 * quarter]
    padded[padded_half : padded_half + quarter] = weight[quarter : 2 * quarter]
    padded[padded_half + quarter : padded_half + 2 * quarter] = weight[3 * quarter :]
    return nn.Parameter(padded, requires_grad=False)


def _padded_rms_norm(
    hidden_states: torch.Tensor, weight, eps: float, orig_head_dim: int | None
) -> torch.Tensor:
    """RMSNorm in the storage dtype, with the variance rescaled so zero padding lanes do not
    deflate it (`None`: the input is not padded).

    Not promoted to fp32 like stock: the round trip changes the stick tiling of the result
    and the eager elementwise op after it fails to lower. `rsqrt` because `pow(x, -0.5)`
    has no lowering.
    """
    dtype = hidden_states.dtype
    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    if orig_head_dim is not None and orig_head_dim != hidden_states.shape[-1]:
        variance = variance * (hidden_states.shape[-1] / orig_head_dim)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight is not None:
        hidden_states = hidden_states * weight.to(dtype)
    return hidden_states


def _padded_head_dim(orig_head_dim: int) -> int:
    """Two sticks, not one: rope rotates each half independently, so both halves have to
    be stick-aligned."""
    return align_up(orig_head_dim, 2 * STICK)


def _pad_mlp(layer, orig_intermediate: int, padded_intermediate: int) -> None:
    """Gate/up gain zero output rows and down matching zero input columns."""
    if padded_intermediate == orig_intermediate:
        return
    mlp = layer.mlp
    if getattr(mlp, "_spyre_padded_intermediate", None) == padded_intermediate:
        return
    for name in ("gate_proj", "up_proj", "down_proj"):
        _assert_clipping_preserves_zero(getattr(mlp, name), f"vision mlp {name}")
    device = mlp.gate_proj.linear.weight.device
    for name in ("gate_proj", "up_proj"):
        proj = getattr(mlp, name)
        proj.linear = _pad_proj_output_simple(
            _as_plain_linear(proj.linear), 1, orig_intermediate, padded_intermediate
        ).to(device)
    mlp.down_proj.linear = _pad_proj_input_simple(
        _as_plain_linear(mlp.down_proj.linear), 1, orig_intermediate, padded_intermediate
    ).to(device)
    mlp._spyre_padded_intermediate = padded_intermediate


def _prepare_attention(attn, num_heads: int, orig_head_dim: int, padded_head_dim: int) -> None:
    if getattr(attn, "_spyre_padded_head_dim", None) == padded_head_dim:
        return
    if attn.v_norm.with_scale:
        raise NotImplementedError(
            "Scaled Gemma 4 vision V normalization is not supported on Spyre."
        )
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _assert_clipping_preserves_zero(getattr(attn, name), f"vision attention {name}")
    device = attn.q_norm.weight.device
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        proj = getattr(attn, name)
        proj.linear = _as_plain_linear(proj.linear)
    attn.q_proj.linear = _pad_qk_linear(attn.q_proj, num_heads, orig_head_dim, padded_head_dim).to(
        device
    )
    attn.k_proj.linear = _pad_qk_linear(attn.k_proj, num_heads, orig_head_dim, padded_head_dim).to(
        device
    )
    attn.v_proj.linear = _pad_proj_output_simple(
        attn.v_proj.linear, num_heads, orig_head_dim, padded_head_dim
    ).to(device)
    attn.o_proj.linear = _pad_proj_input_simple(
        attn.o_proj.linear, num_heads, orig_head_dim, padded_head_dim
    ).to(device)
    attn.q_norm.weight = nn.Parameter(
        _pad_norm_weight(attn.q_norm, orig_head_dim, padded_head_dim).to(device),
        requires_grad=False,
    )
    attn.k_norm.weight = nn.Parameter(
        _pad_norm_weight(attn.k_norm, orig_head_dim, padded_head_dim).to(device),
        requires_grad=False,
    )
    # At the projection dtype: transformers builds the norm weights fp32, which would
    # promote every rope matmul.
    attn.register_buffer(
        "spyre_rope_swap",
        rope_perm_matrix("half", padded_head_dim, device, attn.q_proj.linear.weight.dtype),
        persistent=False,
    )
    attn._spyre_padded_head_dim = padded_head_dim


def pad_vision_weights(encoder) -> None:
    """Pad the encoder's attention projections, norms and MLP width to the stick."""
    config = encoder.config
    padded_head_dim = _padded_head_dim(config.head_dim)
    padded_intermediate = align_up(config.intermediate_size)
    for layer in encoder.layers[: config.num_hidden_layers]:
        _prepare_attention(
            layer.self_attn, config.num_attention_heads, config.head_dim, padded_head_dim
        )
        _pad_mlp(layer, config.intermediate_size, padded_intermediate)
    logger.info_once(
        "Spyre: padded the Gemma 4 vision tower's head dim to %d and its MLP width to %d.",
        padded_head_dim,
        padded_intermediate,
    )


# --- RoPE ---


def _fp32_inv_freq(rotary_emb, config) -> tuple[torch.Tensor, float]:
    """Recomputed in fp32: `model.to(...)` downcasts the module's `inv_freq` buffer, and
    position times a 16-bit frequency loses the angle. The initializer is resolved off
    the rotary module first: transformers renames this rope type after 5.16.1 and the
    new initializer is a class-local staticmethod absent from `ROPE_INIT_FUNCTIONS`."""
    params = getattr(config, "rope_parameters", None) or {}
    rope_type = params.get("rope_type", "default")
    init_fn = getattr(rotary_emb, f"compute_{rope_type}_rope_parameters", None)
    if init_fn is None:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        init_fn = ROPE_INIT_FUNCTIONS[rope_type]
    inv_freq, attention_scaling = init_fn(config)
    return inv_freq.float(), float(attention_scaling)


def gemma4_rope_axis_tables(
    inv_freq: torch.Tensor,
    axis: int,
    orig_head_dim: int,
    padded_head_dim: int,
    rows: int,
    attention_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 `[rows, padded_head_dim]` cos and signed-sin tables for one position axis.

    Gemma 4 vision rotates its two position axes independently, each with its own
    `rotate_half`. `_pad_qk_linear` packs the head as `[X, Y, 0 | X, Y, 0]`, so this axis's
    angle goes into its quarter of both halves, with `rotate_half`'s sign flip in `sin`;
    the other lanes are the identity rotation (cos 1, sin 0), which is what lets the two
    axes combine as `cos_x * cos_y` and `sin_x + sin_y`.
    """
    quarter = orig_head_dim // 4
    half = padded_head_dim // 2
    angles = torch.arange(rows, dtype=torch.float32)[:, None] * inv_freq.float()[None, :]
    cos = torch.ones(rows, padded_head_dim)
    sin = torch.zeros(rows, padded_head_dim)
    start = axis * quarter
    for base, sign in ((0, -1.0), (half, 1.0)):
        cos[:, base + start : base + start + quarter] = angles.cos() * attention_scaling
        sin[:, base + start : base + start + quarter] = sign * angles.sin() * attention_scaling
    return cos, sin


# --- the swapped-in classes ---


class SpyreGemma4RMSNorm(modeling_gemma4.Gemma4RMSNorm):
    """Upstream's norm without the fp32 round trip (see `_padded_rms_norm`)."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.with_scale else None
        return _padded_rms_norm(hidden_states, weight, self.eps, None)


class SpyreGemma4VisionAttention(modeling_gemma4.Gemma4VisionAttention):
    """Upstream's vision attention over the padded head dim, with RoPE as a matmul and
    the stick-aligned SDPA. `pad_vision_weights` has to run before the first call."""

    spyre_rope_swap: torch.Tensor
    _spyre_padded_head_dim: int

    def forward(  # ty: ignore[invalid-method-override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del position_ids, kwargs
        cos, sin = position_embeddings
        bsz, seq_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        orig_head_dim = self.head_dim
        padded_head_dim = self._spyre_padded_head_dim

        q = self.q_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
        q = _padded_rms_norm(q, self.q_norm.weight, self.q_norm.eps, orig_head_dim)
        q = rope_rotate_matmul(q, cos, sin, self.spyre_rope_swap).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
        k = _padded_rms_norm(k, self.k_norm.weight, self.k_norm.eps, orig_head_dim)
        k = rope_rotate_matmul(k, cos, sin, self.spyre_rope_swap).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
        v = _padded_rms_norm(v, None, self.v_norm.eps, orig_head_dim).transpose(1, 2)

        out = padded_sdpa(q, k, v, attention_mask, scale=float(self.scaling))
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out), None


def swap_vision_classes(encoder: nn.Module) -> None:
    """Swap each layer's attention and norms to the Spyre classes in place, keeping the
    modules, parameters and names vLLM already built and will load."""
    for layer in cast(nn.ModuleList, encoder.layers):
        layer.self_attn.__class__ = SpyreGemma4VisionAttention
        for module in layer.modules():
            if type(module) is modeling_gemma4.Gemma4RMSNorm:
                module.__class__ = SpyreGemma4RMSNorm


def _unbind(t: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    return list(t.unbind(0)) if isinstance(t, torch.Tensor) else list(t)


def _images(mm_kwargs: dict[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """`([patches, pixels], [patches, 2])` per image, whether batched or listed."""
    pixel_values = mm_kwargs.get("pixel_values")
    if pixel_values is None:
        return []
    return list(zip(_unbind(pixel_values), _unbind(mm_kwargs["pixel_position_ids"]), strict=True))


def _valid_patches(pixel_values: torch.Tensor, positions: torch.Tensor):
    """The image's real patches and positions: the processor pads with (-1, -1) after them."""
    valid = ~(positions == -1).all(dim=-1)
    return pixel_values[valid], positions[valid]


# The mixin's protocol methods shadow upstream's on purpose; their signatures drop
# upstream's `**kwargs` and defaults.
class SpyreGemma4ForConditionalGeneration(  # ty: ignore[invalid-method-override]
    SpyreVisionEncoderMixin, Gemma4ForConditionalGeneration
):
    """Upstream Gemma 4 with its vision encoder compiled per token budget."""

    vision_modalities = ("image",)
    # The one-hot image matrix is 64 wide, and its last column belongs to the padding.
    vision_max_items_per_call = STICK - 1

    spyre_cos_x: GatherTable
    spyre_sin_x: GatherTable
    spyre_cos_y: GatherTable
    spyre_sin_y: GatherTable

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision()

    def _init_spyre_vision(self) -> None:
        """Set the mixin's attributes, swap in the Spyre classes and build the RoPE tables.
        Separate from ``__init__`` so tests can run it on upstream modules."""
        cfg = self.config.vision_config
        self._pool = cfg.pooling_kernel_size
        self.patches_per_token = self._pool**2
        self.out_hidden_size = self.config.text_config.hidden_size
        self._patch_width = align_up(3 * cfg.patch_size**2)
        self._padded_head_dim = _padded_head_dim(cfg.head_dim)
        self.vision_buffers = {
            "patches": VisionBuffer(self._patch_width),
            "segments": VisionBuffer(STICK, pad=F.one_hot(torch.tensor(STICK - 1), STICK).half()),
            "pos_x": VisionBuffer(None, pad=0, dtype=torch.int64),
            "pos_y": VisionBuffer(None, pad=0, dtype=torch.int64),
        }
        # With every image limit at 0, upstream leaves a placeholder instead of the tower.
        tower = self.vision_tower
        if not isinstance(tower, modeling_gemma4.Gemma4VisionModel):
            return
        if cfg.num_key_value_heads != cfg.num_attention_heads:
            raise NotImplementedError("Gemma 4 vision GQA is not supported on Spyre.")
        swap_vision_classes(tower.encoder)
        inv_freq, scaling = _fp32_inv_freq(tower.encoder.rotary_emb, cfg)
        for axis, name in enumerate("xy"):
            cos, sin = gemma4_rope_axis_tables(
                inv_freq,
                axis,
                cfg.head_dim,
                self._padded_head_dim,
                cfg.position_embedding_size,
                scaling,
            )
            setattr(self, f"spyre_cos_{name}", GatherTable(cos.half()))
            setattr(self, f"spyre_sin_{name}", GatherTable(sin.half()))

    def process_weights_after_loading(self) -> None:
        """Pad the tower's weights and reshape the embedding tables for the vision graphs.
        Runs on the host, before the runner moves the model to the card."""
        tower = self.vision_tower
        if not isinstance(tower, modeling_gemma4.Gemma4VisionModel):
            return
        pad_vision_weights(tower.encoder)
        embedder = tower.patch_embedder
        weight = _linear_weight(embedder.input_proj)
        weight_t = F.pad(weight, (0, self._patch_width - weight.shape[1])).t().contiguous()
        embedder.register_buffer("spyre_patch_weight_t", weight_t, persistent=False)
        # transformers creates its own parameters and buffers in fp32 and the loader keeps
        # that dtype, so everything the graphs read follows the projections' dtype.
        dtype = weight.dtype
        table = _host(embedder.position_embedding_table).to(dtype)
        embedder.spyre_pos_x = GatherTable(table[0].clone())
        embedder.spyre_pos_y = GatherTable(table[1].clone())
        if tower.config.standardize:
            tower.std_bias = tower.std_bias.to(dtype)
            tower.std_scale = tower.std_scale.to(dtype)
        for module in tower.encoder.modules():
            if isinstance(module, SpyreGemma4RMSNorm) and module.with_scale:
                module.weight = nn.Parameter(_host(module.weight).to(dtype), requires_grad=False)

    # --- SpyreVisionEncoderMixin hooks ---

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        return [
            _valid_patches(pv, pp)[1].shape[0] // self.patches_per_token
            for pv, pp in _images(mm_kwargs)
        ]

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Offset-major patches of every image, normalized as the patch embedder does, with
        each patch's one-hot image id and its (x, y) position."""
        pool = self._pool
        patches, pos_x, pos_y, seg = [], [], [], []
        for i, (pv, pp) in enumerate(_images(mm_kwargs)):
            pv, pp = _valid_patches(pv, pp)
            wp, hp = int(pp[:, 0].max()) + 1, int(pp[:, 1].max()) + 1
            if pv.shape[0] != hp * wp or hp % pool or wp % pool:
                raise ValueError(
                    f"image {i}: {pv.shape[0]} patches on a {hp}x{wp} grid is not whole "
                    f"{pool}x{pool} pooling cells"
                )
            grid = (2.0 * (convert(pv, device="cpu").float() - 0.5)).reshape(hp, wp, -1)
            patches.append(offset_major(grid, pool))
            coords = convert(pp, device="cpu").reshape(hp, wp, 2)
            pos_x.append(offset_major(coords[..., 0], pool))
            pos_y.append(offset_major(coords[..., 1], pool))
            seg.append(torch.full((pool * pool, (hp // pool) * (wp // pool)), i))
        flat = torch.cat(patches, dim=1).flatten(0, 1).to(torch.float16)
        return {
            "patches": F.pad(flat, (0, self._patch_width - flat.shape[1])),
            "segments": F.one_hot(torch.cat(seg, dim=1).flatten(), STICK).to(torch.float16),
            "pos_x": torch.cat(pos_x, dim=1).flatten().to(torch.int64),
            "pos_y": torch.cat(pos_y, dim=1).flatten().to(torch.int64),
        }

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        embedder = self.vision_tower.patch_embedder
        pos_x, pos_y = values["pos_x"], values["pos_y"]
        x = torch.matmul(values["patches"], cast(torch.Tensor, embedder.spyre_patch_weight_t))
        table_x = cast(GatherTable, embedder.spyre_pos_x)
        table_y = cast(GatherTable, embedder.spyre_pos_y)
        x = x + table_x.gather(pos_x) + table_y.gather(pos_y)
        # Same-image mask as a matmul: Spyre cannot lay out a column-minus-row broadcast.
        segments = values["segments"]
        mask = ((1.0 - torch.matmul(segments, segments.t())) * _MASK_MIN)[None, None]
        cos = (self.spyre_cos_x.gather(pos_x) * self.spyre_cos_y.gather(pos_y))[None, :, None]
        sin = (self.spyre_sin_x.gather(pos_x) + self.spyre_sin_y.gather(pos_y))[None, :, None]
        return x.unsqueeze(0), {"position_embeddings": (cos, sin), "attention_mask": mask}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_tower.encoder.layers

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        tower = self.vision_tower
        # One [tokens, D] group per pooling offset: the 3x3 average is their mean.
        x = x.squeeze(0).reshape(self.patches_per_token, -1, x.shape[-1]).mean(dim=0)
        x = x * tower.pooler.root_hidden_size
        if tower.config.standardize:
            x = (x - tower.std_bias) * tower.std_scale
        # `embed_vision`'s own RMSNorm round-trips through fp32, whose layout the device
        # cannot rescale back to fp16; the projection linear is used as is.
        embed = self.embed_vision
        x = _padded_rms_norm(x, None, embed.eps, None)
        x, _ = embed.embedding_projection(x)
        return x
