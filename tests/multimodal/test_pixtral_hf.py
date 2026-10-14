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

"""Tests for `spyre_inference/multimodal/pixtral_hf.py` and its dispatch."""

import pytest
import torch

pixtral = pytest.importorskip("vllm.model_executor.models.pixtral")
modeling_pixtral = pytest.importorskip("transformers.models.pixtral.modeling_pixtral")

HIDDEN_SIZE = 256


@pytest.fixture(autouse=True)
def restore_block_mask(monkeypatch):
    """The mask patch mutates the shared transformers module; undo it after each test."""
    monkeypatch.setattr(
        modeling_pixtral,
        "generate_block_attention_mask",
        modeling_pixtral.generate_block_attention_mask,
    )


class PixtralHFVisionModel(torch.nn.Module):
    """Dispatch keys on the tower's class name."""

    def __init__(self) -> None:
        super().__init__()
        self.ln_pre = torch.nn.RMSNorm(HIDDEN_SIZE)


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.vision_tower = PixtralHFVisionModel()
    return model


def test_hf_tower_is_dispatched_to_its_patches():
    from spyre_inference.multimodal import apply_multimodal_patches
    from spyre_inference.multimodal.pixtral_hf import _DefaultLayoutNorm

    model = _model()
    apply_multimodal_patches(model, torch.device("cpu"))

    assert isinstance(model.vision_tower.ln_pre, _DefaultLayoutNorm)
    assert modeling_pixtral.generate_block_attention_mask._spyre_patched is True


def test_patches_are_idempotent():
    from spyre_inference.multimodal import pixtral_hf

    model = _model()
    pixtral_hf.apply(model, torch.device("cpu"))
    norm, mask = model.vision_tower.ln_pre, modeling_pixtral.generate_block_attention_mask
    pixtral_hf.apply(model, torch.device("cpu"))
    assert model.vision_tower.ln_pre is norm
    assert modeling_pixtral.generate_block_attention_mask is mask


def test_pre_transformer_norm_patch_is_transparent_on_cpu():
    from spyre_inference.multimodal.pixtral_hf import patch_pre_transformer_norm

    model = _model()
    x = torch.randn(1, 49, HIDDEN_SIZE)
    expected = model.vision_tower.ln_pre(x)
    patch_pre_transformer_norm(model)
    torch.testing.assert_close(model.vision_tower.ln_pre(x), expected)


@pytest.mark.parametrize("patch_embeds_list", [[16], [16, 16], [9, 16, 25]])
def test_cpu_block_mask_matches_upstream(patch_embeds_list):
    """A CPU tensor takes the pass-through branch; the device branch only swaps in a CPU
    stand-in of the same shape and dtype."""
    from spyre_inference.multimodal.pixtral_hf import patch_block_attention_mask

    seq = sum(patch_embeds_list)
    embeds = torch.zeros(1, seq, 8, dtype=torch.float16)
    want = modeling_pixtral.generate_block_attention_mask(patch_embeds_list, embeds)
    patch_block_attention_mask()
    got = modeling_pixtral.generate_block_attention_mask(patch_embeds_list, embeds)
    assert torch.equal(got, want)


def test_apply_rejects_an_xformers_install(monkeypatch):
    """With xformers importable, upstream hands the tower a `BlockDiagonalMask` and the
    mask patch silently stops applying; fail at patch time, not inside the SDPA."""
    from spyre_inference.multimodal import pixtral_hf

    monkeypatch.setattr(pixtral, "USE_XFORMERS_OPS", True, raising=False)
    with pytest.raises(RuntimeError, match="xformers"):
        pixtral_hf.apply(_model(), torch.device("cpu"))


def test_native_pixtral_is_not_patched():
    """The mixin model adapts its own tower; the shared transformers mask stays upstream's."""
    from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin
    from spyre_inference.multimodal import apply_multimodal_patches

    class _Native(SpyreVisionEncoderMixin, torch.nn.Module):
        pass

    model = _Native()
    model.vision_tower = PixtralHFVisionModel()
    orig = modeling_pixtral.generate_block_attention_mask
    apply_multimodal_patches(model, torch.device("cpu"))
    assert modeling_pixtral.generate_block_attention_mask is orig
