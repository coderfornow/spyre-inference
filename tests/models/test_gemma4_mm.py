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

"""``SpyreGemma4ForConditionalGeneration``'s vision pipeline against upstream Gemma 4.

The pipeline strips the processor's padding, reorders patches, pads the head dim and the
MLP, gathers the position and RoPE tables and pools by a mean; these tests pin each of
those to the upstream result on CPU. The on-card twins at the bottom skip without a device.
"""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
configuration_gemma4 = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
gemma4_mm = pytest.importorskip("vllm.model_executor.models.gemma4_mm")

pytestmark = [pytest.mark.gemma4_vision]

# 26B-A4B's real vision head_dim: 72 is neither a 64-multiple nor evenly halvable onto
# the stick, which is the whole reason the padding exists.
ORIG_HEAD_DIM = 72
PADDED_HEAD_DIM = 128
NUM_HEADS = 2
HIDDEN = NUM_HEADS * ORIG_HEAD_DIM
INTERMEDIATE = 100  # off the stick, so the MLP padding is live
POOL = 3
PATCH = 16
POSITIONS = 64
TEXT_HIDDEN = 64


@pytest.fixture(autouse=True)
def _engine_ir_dispatch():
    """Trace vLLM IR ops such as ``RMSNorm`` inline, as the worker does under
    STOCK_TORCH_COMPILE; the wrapped op fails a layout check the engine never reaches."""
    import vllm.ir

    with vllm.ir.enable_torch_wrap(False):
        yield


def _vision_config(num_layers: int = 2, **overrides):
    kwargs = dict(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=num_layers,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=ORIG_HEAD_DIM,
        pooling_kernel_size=POOL,
        patch_size=PATCH,
        position_embedding_size=POSITIONS,
        standardize=True,
        use_clipped_linears=False,
    )
    kwargs.update(overrides)
    return configuration_gemma4.Gemma4VisionConfig(**kwargs)


def _finish_weight_loading(module: nn.Module) -> None:
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _upstream_modules(config) -> nn.ModuleDict:
    """The tower and projector of ``Gemma4ForConditionalGeneration``, seeded.

    Unfinished on purpose: the Spyre linears transpose `weight` when loading finishes,
    so each side copies raw weights first and finishes on its own.
    """
    torch.manual_seed(0)
    text_config = SimpleNamespace(hidden_size=TEXT_HIDDEN)
    mods = nn.ModuleDict(
        {
            "vision_tower": modeling_gemma4.Gemma4VisionModel(config),
            "embed_vision": gemma4_mm.Gemma4MultimodalEmbedder(config, text_config),
        }
    )
    for p in mods.parameters():
        p.data.normal_(std=0.05)
    for name, p in mods.named_parameters():
        if name.endswith("norm.weight"):
            p.data.add_(1.0)
    tower = mods["vision_tower"]
    tower.std_bias.normal_(std=0.1)
    tower.std_scale.normal_(std=0.1).add_(1.0)
    return mods


def _grid_image(hp: int, wp: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One image as the processor emits it before padding: row-major patches in [0, 1]
    and their (x, y) positions."""
    patches = torch.rand(hp * wp, 3 * PATCH**2)
    grid = torch.meshgrid(torch.arange(wp), torch.arange(hp), indexing="xy")
    positions = torch.stack(grid, dim=-1).reshape(hp * wp, 2)
    return patches, positions


def _images(num_images: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(1)
    # 54 + 18 = 72 patches = 8 tokens: off the stick, so padding slots are live.
    return [_grid_image(6, 9), _grid_image(3, 6)][:num_images]


def _processor_batch(images, max_patches: int = 90) -> dict[str, torch.Tensor]:
    """``pixel_values`` / ``pixel_position_ids`` padded as the processor pads them."""
    pv = torch.zeros(len(images), max_patches, 3 * PATCH**2)
    pp = torch.full((len(images), max_patches, 2), -1, dtype=torch.long)
    for i, (patches, positions) in enumerate(images):
        pv[i, : patches.shape[0]] = patches
        pp[i, : positions.shape[0]] = positions
    return {"pixel_values": pv, "pixel_position_ids": pp}


def _reference(config, images) -> list[torch.Tensor]:
    """Upstream ``Gemma4ForConditionalGeneration._process_image_input``, one image at a time."""
    mods = _upstream_modules(config)
    _finish_weight_loading(mods)
    out = []
    with torch.inference_mode():
        for patches, positions in images:
            hidden = mods["vision_tower"](patches[None], positions[None]).last_hidden_state
            out.append(mods["embed_vision"](hidden.unsqueeze(0)).squeeze(0))
    return out


def _spyre_model(config, dtype=torch.float16):
    """A ``SpyreGemma4ForConditionalGeneration`` around the same (seeded) upstream
    modules as ``_reference``, with the steps weight loading performs."""
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    model = SpyreGemma4ForConditionalGeneration.__new__(SpyreGemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=config, text_config=SimpleNamespace(hidden_size=TEXT_HIDDEN)
    )
    for name, mod in _upstream_modules(config).items():
        setattr(model, name, mod)
    model._init_spyre_vision()
    # The loader's finish steps, in its order: per-layer first, then the model hook.
    _finish_weight_loading(model)
    model.process_weights_after_loading()
    return model.to(dtype)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_images", [1, 2])
def test_pipeline_matches_upstream(tp_group, num_images):
    config = _vision_config()
    images = _images(num_images)

    expected = _reference(config, images)
    with torch.inference_mode():
        actual = _spyre_model(config).embed_multimodal(**_processor_batch(images))

    assert [t.shape for t in actual] == [t.shape for t in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.device.type == "cpu"
        assert _cosine(got, want) > 0.999


def test_listed_images_match_batched_images(tp_group):
    """The processor hands a list of per-image tensors or one batched tensor."""
    config = _vision_config(num_layers=1)
    model = _spyre_model(config)
    images = _images()
    batch = _processor_batch(images)
    listed = {k: list(v.unbind(0)) for k, v in batch.items()}

    with torch.inference_mode():
        from_batch = model.embed_multimodal(**batch)
        from_list = model.embed_multimodal(**listed)

    for a, b in zip(from_batch, from_list, strict=True):
        torch.testing.assert_close(a, b)


def test_item_tokens_strip_the_processor_padding(tp_group):
    config = _vision_config(num_layers=1)
    model = _spyre_model(config)
    images = _images()
    assert model.vision_item_tokens(_processor_batch(images, max_patches=180)) == [6, 2]


def test_a_grid_that_is_not_whole_pooling_cells_is_refused(tp_group):
    model = _spyre_model(_vision_config(num_layers=1))
    batch = _processor_batch([_grid_image(4, 6)])
    with pytest.raises(ValueError, match="pooling cells"):
        model.vision_inputs(batch)


def test_padding_does_not_leak_into_images(tp_group):
    """Bucket padding must not change any image's embedding: padding slots have their own
    segment, and each offset block is padded separately."""
    config = _vision_config(num_layers=1)
    model = _spyre_model(config)
    batch = _processor_batch(_images())
    values = model.prepare_encoder_cudagraph_replay_buffers(batch, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics

    def run(num_patches):
        bufs = model._vision_empty(num_patches)
        for key, src in values.items():
            padding[key](bufs[key], src)
        return model.encoder_cudagraph_forward(bufs)

    with torch.inference_mode():
        tight, loose = run(576), run(1152)
    n = sum(model.vision_item_tokens(batch))
    torch.testing.assert_close(tight[:n], loose[:n], atol=1e-3, rtol=1e-3)


def test_offset_major_mean_matches_upstream_pooler():
    """Offset-major patches averaged over their groups equal the position-indexed pooler,
    in the same token order."""
    from spyre_inference.models._vision_encoder import offset_major

    torch.manual_seed(2)
    hp, wp, d = 6, 9, 16
    hidden = torch.randn(hp * wp, d)
    _, positions = _grid_image(hp, wp)
    pooler = modeling_gemma4.Gemma4VisionPooler(_vision_config(num_layers=1))
    padding = torch.zeros(1, hp * wp, dtype=torch.bool)
    expected, mask = pooler(hidden[None], positions[None], padding, (hp * wp) // POOL**2)
    assert mask.all()

    blocks = offset_major(hidden.reshape(hp, wp, d), POOL)
    got = blocks.mean(0) * pooler.root_hidden_size
    torch.testing.assert_close(got, expected.squeeze(0), atol=1e-5, rtol=1e-5)


def test_rope_tables_match_upstream_multidimensional_rope():
    """Two per-axis gather tables, combined as cos_x*cos_y and sin_x+sin_y on the padded
    head layout, rotate like stock ``apply_multidimensional_rope`` on the real channels."""
    from spyre_inference.models.gemma4_mm import (
        _fp32_inv_freq,
        _padded_head_dim,
        gemma4_rope_axis_tables,
    )
    from spyre_inference.multimodal.utils import rope_perm_matrix, rope_rotate_matmul

    config = _vision_config(num_layers=1)
    rotary = modeling_gemma4.Gemma4VisionRotaryEmbedding(config)
    torch.manual_seed(3)
    seq = 40
    positions = torch.randint(0, POSITIONS, (1, seq, 2))
    x = torch.randn(1, seq, NUM_HEADS, ORIG_HEAD_DIM)
    cos, sin = rotary(x, positions)
    want = modeling_gemma4.apply_multidimensional_rope(x, cos, sin, positions)

    inv_freq, scaling = _fp32_inv_freq(rotary, config)
    padded = _padded_head_dim(ORIG_HEAD_DIM)
    tables = [
        gemma4_rope_axis_tables(inv_freq, axis, ORIG_HEAD_DIM, padded, POSITIONS, scaling)
        for axis in range(2)
    ]
    px, py = positions[0, :, 0], positions[0, :, 1]
    cos_full = (tables[0][0][px] * tables[1][0][py])[None, :, None]
    sin_full = (tables[0][1][px] + tables[1][1][py])[None, :, None]
    perm = rope_perm_matrix("half", padded, torch.device("cpu"), torch.float32)
    got = rope_rotate_matmul(_pad_quarters(x), cos_full, sin_full, perm)
    torch.testing.assert_close(_unpad_quarters(got), want, atol=1e-5, rtol=1e-5)
    # The padding lanes stay an exact identity rotation of the zero input lanes.
    assert (got[..., 2 * (ORIG_HEAD_DIM // 4) : padded // 2] == 0).all()


def _pad_quarters(x: torch.Tensor) -> torch.Tensor:
    """The `_pad_qk_linear` channel remap, applied to activations instead of weights."""
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    out = torch.zeros((*x.shape[:-1], PADDED_HEAD_DIM), dtype=x.dtype)
    out[..., :quarter] = x[..., :quarter]
    out[..., quarter : 2 * quarter] = x[..., 2 * quarter : 3 * quarter]
    out[..., half : half + quarter] = x[..., quarter : 2 * quarter]
    out[..., half + quarter : half + 2 * quarter] = x[..., 3 * quarter :]
    return out


def _unpad_quarters(padded: torch.Tensor) -> torch.Tensor:
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    out = torch.zeros((*padded.shape[:-1], ORIG_HEAD_DIM), dtype=padded.dtype)
    out[..., :quarter] = padded[..., :quarter]
    out[..., 2 * quarter : 3 * quarter] = padded[..., quarter : 2 * quarter]
    out[..., quarter : 2 * quarter] = padded[..., half : half + quarter]
    out[..., 3 * quarter :] = padded[..., half + quarter : half + 2 * quarter]
    return out


# ---------------------------------------------------------------------------
# Construction, registration, the swaps
# ---------------------------------------------------------------------------


def test_swaps_keep_the_modules_and_their_names(tp_group):
    """The class swap must not rebuild a module: vLLM's linear replacement already ran
    and the loader matches parameters by name."""
    from spyre_inference.models.gemma4_mm import (
        SpyreGemma4RMSNorm,
        SpyreGemma4VisionAttention,
        swap_vision_classes,
    )

    tower = modeling_gemma4.Gemma4VisionModel(_vision_config())
    before = {name: id(m) for name, m in tower.named_modules()}
    keys = list(tower.state_dict())
    swap_vision_classes(tower.encoder)

    assert {name: id(m) for name, m in tower.named_modules()} == before
    assert list(tower.state_dict()) == keys
    for layer in tower.encoder.layers:
        assert type(layer.self_attn) is SpyreGemma4VisionAttention
        norms = [layer.input_layernorm, layer.self_attn.q_norm, layer.self_attn.v_norm]
        assert all(type(n) is SpyreGemma4RMSNorm for n in norms)
    # The projector's norm is vLLM's, not transformers': untouched.
    assert not any(isinstance(m, SpyreGemma4RMSNorm) for m in tower.patch_embedder.modules())


def test_swapped_norm_stays_in_the_input_dtype():
    """transformers builds these weights fp32 and the tower feeds fp16; stock hides the
    promotion behind a trailing `.type_as`, which the device cannot lower."""
    from spyre_inference.models.gemma4_mm import SpyreGemma4RMSNorm

    norm = modeling_gemma4.Gemma4RMSNorm(dim=16, eps=1e-6, with_scale=True)
    assert norm.weight.dtype == torch.float32, "premise of this test changed upstream"
    norm.__class__ = SpyreGemma4RMSNorm
    x = torch.randn(2, 3, 16, dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16
    want = modeling_gemma4.Gemma4RMSNorm.forward(norm, x.float())
    assert _cosine(out, want) > 0.9999


def test_stage_skipped_tower_is_left_alone(tp_group):
    """With every image limit at 0, vLLM leaves a ``StageMissingLayer`` where the tower
    was: nothing to swap, pad or table."""
    from vllm.model_executor.models.utils import StageMissingLayer

    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    config = _vision_config()
    model = SpyreGemma4ForConditionalGeneration.__new__(SpyreGemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=config, text_config=SimpleNamespace(hidden_size=TEXT_HIDDEN)
    )
    model.vision_tower = StageMissingLayer("vision_tower", nn.Module())
    model._init_spyre_vision()
    model.process_weights_after_loading()

    assert model.patches_per_token == POOL**2
    assert set(model.vision_buffers) == {"patches", "segments", "pos_x", "pos_y"}
    assert not hasattr(model, "spyre_cos_x")


def test_registry_resolves_to_spyre_class_and_keeps_processor():
    from spyre_inference.models import spyre_models
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    assert (
        spyre_models()["Gemma4ForConditionalGeneration"]
        == "spyre_inference.models.gemma4_mm:SpyreGemma4ForConditionalGeneration"
    )
    upstream = gemma4_mm.Gemma4ForConditionalGeneration
    assert SpyreGemma4ForConditionalGeneration._processor_factory is upstream._processor_factory


def test_construction_reaches_upstream_init_and_protocol():
    """``super().__init__`` must land on upstream's: a ``Protocol`` base ahead of it in the
    MRO would substitute its placeholder ``__init__`` and skip building the model."""
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    mro = SpyreGemma4ForConditionalGeneration.__mro__[1:]
    first_init = next(cls for cls in mro if "__init__" in cls.__dict__)
    assert first_init is gemma4_mm.Gemma4ForConditionalGeneration
    assert supports_encoder_cudagraph(SpyreGemma4ForConditionalGeneration)
    # Upstream's own protocol methods that the mixin leaves in place.
    cls = SpyreGemma4ForConditionalGeneration
    for name in ("get_input_modality", "select_encoder_cudagraph_items"):
        assert getattr(cls, name) is getattr(gemma4_mm.Gemma4ForConditionalGeneration, name)


def test_upstream_tower_contract_is_unchanged():
    """The swapped classes keep upstream's call contract; diff this on a transformers bump."""
    params = list(inspect.signature(modeling_gemma4.Gemma4VisionAttention.forward).parameters)
    assert params[:4] == ["self", "hidden_states", "position_embeddings", "attention_mask"]
    layer_src = inspect.getsource(modeling_gemma4.Gemma4VisionEncoderLayer.forward)
    assert "position_embeddings=position_embeddings" in layer_src
    assert "attention_mask=attention_mask" in layer_src
    for name in ("patch_embedder", "encoder", "pooler"):
        assert hasattr(modeling_gemma4.Gemma4VisionModel(_vision_config(num_layers=1)), name)


@pytest.mark.parametrize(
    ("limits", "video"),
    [
        ({}, 0),
        ({"image": 2}, 0),
        ({"image": 1, "video": 0}, 0),
        ({"video": {"count": 0}}, {"count": 0}),
    ],
)
def test_video_limit_defaults_to_zero(monkeypatch, limits, video):
    from spyre_inference.models import gemma4

    monkeypatch.setattr(
        gemma4, "_gemma4_hf_config", lambda _: SimpleNamespace(model_type="gemma4", vision_config=1)
    )
    args = SimpleNamespace(limit_mm_per_prompt=dict(limits))
    gemma4.default_mm_limits(args)
    assert args.limit_mm_per_prompt["video"] == video
    assert all(args.limit_mm_per_prompt[k] == v for k, v in limits.items() if k != "video")


@pytest.mark.parametrize("video", [1, {"count": 2}])
def test_explicit_video_limit_is_refused(monkeypatch, video):
    from spyre_inference.models import gemma4

    monkeypatch.setattr(
        gemma4, "_gemma4_hf_config", lambda _: SimpleNamespace(model_type="gemma4", vision_config=1)
    )
    with pytest.raises(NotImplementedError, match="video"):
        gemma4.default_mm_limits(SimpleNamespace(limit_mm_per_prompt={"video": video}))


def test_text_only_and_other_checkpoints_keep_their_limits(monkeypatch):
    from spyre_inference.models import gemma4

    for cfg in (None, SimpleNamespace(model_type="gemma4", vision_config=None)):
        monkeypatch.setattr(gemma4, "_gemma4_hf_config", lambda _, cfg=cfg: cfg)
        args = SimpleNamespace(limit_mm_per_prompt={"video": 3})
        gemma4.default_mm_limits(args)
        assert args.limit_mm_per_prompt == {"video": 3}


# ---------------------------------------------------------------------------
# Head-dim and MLP padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("orig", "expected"),
    [(64, 128), (72, 128), (128, 128), (129, 256), (256, 256)],
)
def test_padded_head_dim_rounds_to_a_double_stick(orig, expected):
    """Rope needs each half on a whole stick, so head_dim pads to 2*64, not 64."""
    from spyre_inference.models.gemma4_mm import _padded_head_dim

    assert _padded_head_dim(orig) == expected


def test_pad_qk_linear_preserves_every_original_channel():
    """The quarter-interleave must be a permutation of the real channels into the
    padded layout: no value dropped, no value duplicated."""
    from spyre_inference.models.gemma4_mm import _pad_qk_linear

    torch.manual_seed(0)
    in_features = 32
    linear = nn.Linear(in_features, NUM_HEADS * ORIG_HEAD_DIM, bias=False)
    linear.weight.data.normal_()
    proj = SimpleNamespace(linear=linear)
    padded = _pad_qk_linear(proj, NUM_HEADS, ORIG_HEAD_DIM, PADDED_HEAD_DIM)
    assert padded.weight.shape == (NUM_HEADS * PADDED_HEAD_DIM, in_features)

    orig_w = linear.weight.detach().view(NUM_HEADS, ORIG_HEAD_DIM, in_features)
    new_w = padded.weight.detach().view(NUM_HEADS, PADDED_HEAD_DIM, in_features)
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    torch.testing.assert_close(new_w[:, :quarter], orig_w[:, :quarter])
    torch.testing.assert_close(
        new_w[:, quarter : 2 * quarter], orig_w[:, 2 * quarter : 3 * quarter]
    )
    torch.testing.assert_close(new_w[:, half : half + quarter], orig_w[:, quarter : 2 * quarter])
    torch.testing.assert_close(
        new_w[:, half + quarter : half + 2 * quarter], orig_w[:, 3 * quarter :]
    )
    assert torch.all(new_w[:, 2 * quarter : half] == 0)
    assert torch.all(new_w[:, half + 2 * quarter :] == 0)


def test_pad_norm_weight_fills_padding_lanes_with_ones():
    from spyre_inference.models.gemma4_mm import _pad_norm_weight

    norm = modeling_gemma4.Gemma4RMSNorm(dim=ORIG_HEAD_DIM, eps=1e-6, with_scale=True)
    norm.weight.data.normal_()
    padded = _pad_norm_weight(norm, ORIG_HEAD_DIM, PADDED_HEAD_DIM)
    assert padded.shape == (PADDED_HEAD_DIM,)
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    assert torch.all(padded[2 * quarter : half] == 1.0)
    assert torch.all(padded[half + 2 * quarter :] == 1.0)


def test_pad_mlp_is_numerically_transparent():
    """The zero lanes contribute nothing: `gelu(0) * 0 == 0`, and the matching zero
    down-projection columns ignore whatever lands there. Idempotent across loads."""
    from spyre_inference.models.gemma4_mm import _pad_mlp

    torch.manual_seed(0)
    orig, padded = 200, 256
    config = _vision_config(num_layers=1, hidden_size=128, head_dim=64, intermediate_size=orig)
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0).float()
    x = torch.randn(1, 5, 128)
    want = layer.mlp(x)
    _pad_mlp(layer, orig, padded)
    got = layer.mlp(x)
    assert layer.mlp.gate_proj.linear.out_features == padded
    assert layer.mlp.down_proj.linear.in_features == padded
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    _pad_mlp(layer, orig, padded)
    assert layer.mlp.gate_proj.linear.out_features == padded


def test_pad_mlp_is_a_noop_when_already_stick_aligned():
    from spyre_inference.models.gemma4_mm import _pad_mlp

    config = _vision_config(num_layers=1, hidden_size=128, head_dim=64, intermediate_size=256)
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0)
    before = layer.mlp.gate_proj.linear
    _pad_mlp(layer, 256, 256)
    assert layer.mlp.gate_proj.linear is before


def test_padded_rms_norm_ignores_zero_padding_lanes():
    """Normalizing the padded activation must match normalizing the real channels
    alone, which is what the `padded/orig` variance rescale buys."""
    from spyre_inference.models.gemma4_mm import _padded_rms_norm

    torch.manual_seed(0)
    real = torch.randn(1, 4, NUM_HEADS, ORIG_HEAD_DIM)
    got = _unpad_quarters(_padded_rms_norm(_pad_quarters(real), None, 1e-6, ORIG_HEAD_DIM))
    want = real * torch.rsqrt((real * real).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_padded_rms_norm_does_not_promote_via_an_fp32_weight():
    from spyre_inference.models.gemma4_mm import _padded_rms_norm

    x = torch.randn(1, 4, NUM_HEADS, PADDED_HEAD_DIM, dtype=torch.float16)
    out = _padded_rms_norm(x, torch.randn(PADDED_HEAD_DIM), 1e-6, ORIG_HEAD_DIM)
    assert out.dtype == torch.float16


def test_rope_swap_is_a_non_persistent_buffer_at_the_projection_dtype(tp_group):
    """Out of `state_dict`, so weight loading never sees it; fp16 even though transformers
    builds the norm weights fp32, or every rope matmul would promote."""
    model = _spyre_model(_vision_config())
    for layer in model.vision_tower.encoder.layers:
        attn = layer.self_attn
        assert attn.spyre_rope_swap.shape == (PADDED_HEAD_DIM, PADDED_HEAD_DIM)
        assert attn.spyre_rope_swap.dtype == torch.float16
        assert "spyre_rope_swap" not in attn.state_dict()
        assert attn._spyre_padded_head_dim == PADDED_HEAD_DIM


def _clipped_proj(output_min: float, output_max: float):
    config = _vision_config(use_clipped_linears=True)
    proj = modeling_gemma4.Gemma4ClippableLinear(config, 32, NUM_HEADS * ORIG_HEAD_DIM)
    proj.output_min = nn.Buffer(torch.tensor(output_min))
    proj.output_max = nn.Buffer(torch.tensor(output_max))
    return proj


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-6.0, 6.0), (0.0, 6.0), (-6.0, 0.0), (-float("inf"), float("inf"))],
)
def test_clipping_that_includes_zero_is_accepted(lo, hi):
    from spyre_inference.models.gemma4_mm import _assert_clipping_preserves_zero

    _assert_clipping_preserves_zero(_clipped_proj(lo, hi), "test proj")


@pytest.mark.parametrize(("lo", "hi"), [(0.5, 6.0), (-6.0, -0.5)])
def test_clipping_that_excludes_zero_is_rejected(lo, hi):
    from spyre_inference.models.gemma4_mm import _assert_clipping_preserves_zero

    with pytest.raises(NotImplementedError, match="must include zero"):
        _assert_clipping_preserves_zero(_clipped_proj(lo, hi), "test proj")


def test_quantized_projection_is_rejected_rather_than_dequantized():
    from spyre_inference.models.gemma4_mm import _as_plain_linear

    class _FakeQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(32, 8))
            self.bias = None
            self.quant_method = object()

    with pytest.raises(NotImplementedError, match="quantization method"):
        _as_plain_linear(_FakeQuantLinear())


def test_unquantized_vllm_linear_is_bounced_to_a_plain_linear():
    """A Spyre linear stores `Wᵀ`, so the bounce must transpose back to `[out, in]`."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from spyre_inference.models.gemma4_mm import _as_plain_linear

    class _FakeReplicatedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.weight = nn.Parameter(torch.randn(8, 32))  # [in, out]
            self.bias = None
            self.quant_method = UnquantizedLinearMethod()

    src = _FakeReplicatedLinear()
    plain = _as_plain_linear(src)
    assert isinstance(plain, nn.Linear)
    assert plain.weight.shape == (32, 8)
    torch.testing.assert_close(plain.weight, src.weight.t().contiguous())


# ---------------------------------------------------------------------------
# On card
# ---------------------------------------------------------------------------


def _padded_device_inputs(model, batch, token_budget):
    from spyre_inference.custom_ops.utils import convert

    values = model.prepare_encoder_cudagraph_replay_buffers(batch, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    bufs = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget, 1, 1, torch.device("cpu"), torch.float16
    ).values
    for key, src in values.items():
        padding[key](bufs[key], src)
    return {k: convert(v, device="spyre") for k, v in bufs.items()}


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
def test_pipeline_matches_upstream_on_spyre(tp_group):
    """Same comparison with the encoder on the card, uncompiled (the eager path)."""
    config = _vision_config()
    images = _images()

    expected = _reference(config, images)
    with torch.inference_mode():
        actual = _spyre_model(config).to("spyre").embed_multimodal(**_processor_batch(images))

    for got, want in zip(actual, expected, strict=True):
        assert _cosine(got, want) > 0.995


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
@pytest.mark.parametrize("token_budget", [64, 128])
def test_compiled_pipeline_matches_upstream_on_spyre(tp_group, token_budget):
    """The three compiled graphs at one bucket, fed the way the manager feeds them."""
    from spyre_inference.custom_ops.utils import convert

    config = _vision_config()
    images = _images()
    model = _spyre_model(config).to("spyre")
    model.compile_vision_graphs()
    inputs = _padded_device_inputs(model, _processor_batch(images), token_budget)

    expected = torch.cat(_reference(config, images))
    with torch.inference_mode():
        out = model.encoder_cudagraph_forward(inputs)
    actual = convert(out, device="cpu")[: expected.shape[0]]
    assert _cosine(actual, expected) > 0.995


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
def test_compiled_graphs_match_eager_on_the_same_inputs_on_spyre(tp_group):
    """The same padded device buffers through the uncompiled and the compiled pipeline."""
    config = _vision_config()
    batch = _processor_batch(_images())
    model = _spyre_model(config).to("spyre")
    inputs = _padded_device_inputs(model, batch, token_budget=64)
    tokens = sum(model.vision_item_tokens(batch))

    with torch.inference_mode():
        eager = model.encoder_cudagraph_forward(inputs)[:tokens]
        model.compile_vision_graphs()
        compiled = model.encoder_cudagraph_forward(inputs)[:tokens]

    # Cosine only, looser than Pixtral's bar: the fused graph orders the norms' fp16
    # reductions differently from the per-op eager path, and a few of the 512 outputs
    # move by more than any elementwise tolerance that still means something.
    assert _cosine(compiled, eager) > 0.999


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
def test_padding_on_device_weights_matches_padding_on_host():
    """The regression `_host` exists for: composed on device-resident weights, the padding
    helpers' strided slice-assignments lower silently wrong."""
    from spyre_inference.models import gemma4_mm as gm

    config = _vision_config(num_layers=1, intermediate_size=4304)
    torch.manual_seed(0)
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0).half().eval()

    on_host = copy.deepcopy(layer)
    gm._prepare_attention(on_host.self_attn, NUM_HEADS, ORIG_HEAD_DIM, PADDED_HEAD_DIM)
    gm._pad_mlp(on_host, config.intermediate_size, 4352)
    on_host = on_host.to(torch.device("spyre"))

    on_device = copy.deepcopy(layer).to(torch.device("spyre"))
    gm._prepare_attention(on_device.self_attn, NUM_HEADS, ORIG_HEAD_DIM, PADDED_HEAD_DIM)
    gm._pad_mlp(on_device, config.intermediate_size, 4352)

    checked = 0
    for name, want in on_host.named_parameters():
        got = dict(on_device.named_parameters())[name]
        torch.testing.assert_close(
            got.detach().to("cpu").float(),
            want.detach().to("cpu").float(),
            rtol=0,
            atol=torch.finfo(torch.float16).smallest_normal,
            msg=lambda m, name=name: f"{name} differs when padded on device:\n{m}",
        )
        checked += 1
    assert checked
