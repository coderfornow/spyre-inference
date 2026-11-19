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

"""Unit tests for SpyreModelLoader."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from vllm.config import LoadConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.utils import WeightsMapper

from spyre_inference.model_loader import (
    SpyreModelLoader,
    _check_all_weights_loaded,
    _get_host_loaded_param_names,
    _hf_to_vllm_name,
    _param_name_for,
    _UploadStager,
)
from spyre_inference.platform import (
    TorchSpyrePlatform,
    _spyre_loader_supported,
)


def test_spyre_model_loader_registration():
    """Verify that --load-format spyre returns SpyreModelLoader."""
    load_config = LoadConfig(load_format="spyre")
    loader = get_model_loader(load_config)
    assert isinstance(loader, SpyreModelLoader)


def _make_vllm_config(
    tp_size=1,
    quantization=None,
    runner_type="generate",
    load_format="auto",
    transformers_backend=False,
    moe=False,
):
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
        quantization=quantization,
    )
    if runner_type != "generate":
        model_config.runner_type = runner_type
    if transformers_backend:
        model_config.using_transformers_backend = lambda: True
    if moe:
        model_config.hf_text_config.num_experts = 8
    return VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format=load_format),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
    )


def test_platform_defaults_load_format_to_spyre():
    """Positive control: a plain TP=1 text-generation config flips auto → spyre."""
    vllm_config = _make_vllm_config()
    assert vllm_config.load_config.load_format == "spyre"


def test_platform_defaults_load_format_to_spyre_tp2():
    """TP>1 is supported: auto resolves to spyre."""
    vllm_config = _make_vllm_config(tp_size=2)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "spyre"


def test_platform_defaults_load_format_to_spyre_multimodal():
    """Multimodal models are supported: auto resolves to spyre."""
    vllm_config = _make_vllm_config()
    vllm_config.model_config.multimodal_config = object()
    vllm_config.load_config.load_format = "auto"
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "spyre"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantization": "fp8"},  # quantized
        {"runner_type": "pooling"},  # pooling
    ],
    ids=["quantized", "pooling"],
)
def test_platform_load_format_stays_auto(kwargs):
    """load_format stays 'auto' for configs SpyreModelLoader doesn't support."""
    vllm_config = _make_vllm_config(**kwargs)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "auto"


# ---------------------------------------------------------------------------
# _spyre_loader_supported unit tests
# ---------------------------------------------------------------------------


def _make_ns(
    tp_size=1,
    runner_type="generate",
    quantization=None,
    transformers_backend=False,
    moe=False,
    multimodal=False,
):
    """Build a minimal SimpleNamespace config for _spyre_loader_supported tests."""
    model_config = SimpleNamespace(
        runner_type=runner_type,
        quantization=quantization,
        hf_text_config=SimpleNamespace(num_experts=8) if moe else SimpleNamespace(),
        multimodal_config=object() if multimodal else None,
        is_multimodal_model=False,
    )
    if transformers_backend:
        model_config.using_transformers_backend = lambda: True
    parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    return SimpleNamespace(model_config=model_config, parallel_config=parallel_config)


def test_spyre_loader_supported_accepts_valid_config():
    """Positive case: generate, unquantized, dense, non-transformers-backend is supported."""
    assert _spyre_loader_supported(_make_ns())


@pytest.mark.parametrize(
    "kwargs",
    [{"tp_size": 2}, {"tp_size": 4}, {"multimodal": True}, {"tp_size": 2, "multimodal": True}],
    ids=["tp2", "tp4", "multimodal", "tp2-multimodal"],
)
def test_spyre_loader_supported_accepts_tp_and_multimodal(kwargs):
    """TP > 1 and multimodal models are supported by the spyre loader."""
    assert _spyre_loader_supported(_make_ns(**kwargs))


@pytest.mark.parametrize(
    ("kwargs", "description"),
    [
        ({"runner_type": "pooling"}, "pooling runner"),
        ({"runner_type": "unknown_future_runner"}, "unknown runner type"),
        ({"quantization": "fp8"}, "quantized"),
        ({"transformers_backend": True}, "transformers backend"),
        ({"moe": True}, "MoE"),
    ],
)
def test_spyre_loader_supported_rejects_unsupported(kwargs, description):
    """_spyre_loader_supported returns False for each excluded config class."""
    assert not _spyre_loader_supported(_make_ns(**kwargs)), description


def test_platform_load_format_stays_auto_transformers_backend():
    """load_format stays 'auto' for the transformers backend."""
    assert _make_vllm_config(transformers_backend=True).load_config.load_format == "auto"


def test_platform_load_format_stays_auto_moe():
    """load_format stays 'auto' for MoE models."""
    assert _make_vllm_config(moe=True).load_config.load_format == "auto"


def test_explicit_spyre_load_format_rejected_for_unsupported_config():
    """--load-format spyre passed explicitly raises for unsupported configs.

    VllmConfig is built normally (load_format="auto"), then load_format is
    patched to "spyre" directly on the load_config object to bypass pydantic
    validation (which only accepts registered formats at construction time).
    _spyre_loader_supported is mocked to False to simulate an unsupported config.
    """
    import unittest.mock as mock

    import spyre_inference.platform as plat_mod

    vllm_config = _make_vllm_config()
    vllm_config.load_config.load_format = "spyre"
    with (
        mock.patch.object(plat_mod, "_spyre_loader_supported", return_value=False),
        pytest.raises(ValueError, match="--load-format spyre is not supported"),
    ):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


def test_get_host_loaded_param_names(tp_group):
    """Verify VocabParallelEmbedding parameters stay host-loaded."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = VocabParallelEmbedding(10, 10)
            self.dense = nn.Linear(10, 10)

    model = DummyModel()
    host_names = _get_host_loaded_param_names(model)
    assert "embed.weight" in host_names
    assert "dense.weight" not in host_names


def test_check_all_weights_loaded_passes_when_complete():
    model = nn.Sequential(nn.Linear(2, 2))
    _check_all_weights_loaded(model, {"0.weight", "0.bias"})


def test_check_all_weights_loaded_raises_on_missing_param():
    """A parameter with no checkpoint tensor would keep its uninitialized placeholder."""
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="not initialized from checkpoint.*0.bias"):
        _check_all_weights_loaded(model, {"0.weight"})


def test_check_all_weights_loaded_skips_untracked_models():
    """A model whose load_weights returns None cannot be checked; it must not raise."""
    model = nn.Sequential(nn.Linear(2, 2))
    _check_all_weights_loaded(model, None)


@pytest.mark.parametrize("attr", ["_spyre_orig_head_dim", "_spyre_orig_intermediate_size"])
def test_spyre_loader_supported_rejects_padded(attr):
    ns = _make_ns()
    ns.model_config.hf_text_config = SimpleNamespace(**{attr: 16})
    assert not _spyre_loader_supported(ns)


def test_spyre_loader_supported_accepts_unpadded_text_config():
    ns = _make_ns()
    ns.model_config.hf_text_config = SimpleNamespace()
    assert _spyre_loader_supported(ns)


def test_gate_sees_padding_applied_by_the_hook(monkeypatch):
    """The hook must pad before gating, since check_and_update_config runs later."""
    vllm_config = _make_vllm_config()
    vllm_config.load_config.load_format = "auto"

    def fake_pad(cls, config):
        config.model_config.hf_text_config._spyre_orig_head_dim = 16

    monkeypatch.setattr(TorchSpyrePlatform, "_maybe_pad_head_dim", classmethod(fake_pad))
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "auto"


def test_explicit_spyre_load_format_rejected_for_padded_config():
    vllm_config = _make_vllm_config()
    vllm_config.model_config.hf_text_config._spyre_orig_intermediate_size = 160
    vllm_config.load_config.load_format = "spyre"
    with pytest.raises(ValueError, match="--load-format spyre is not supported"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


class _FakeMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(2, 2)
        self.vision_tower = nn.Linear(2, 2)
        self.projector = nn.Linear(2, 2)


_NON_TEXT = {"vision_tower.weight", "vision_tower.bias", "projector.weight", "projector.bias"}


def test_host_loaded_names_include_vision_tower(monkeypatch):
    from spyre_inference.custom_ops import text_backbone

    model = _FakeMultimodal()
    monkeypatch.setattr(
        text_backbone, "_language_backbones", lambda m: [("language_model", m.language_model)]
    )
    assert _get_host_loaded_param_names(model) == _NON_TEXT


def test_non_text_parameter_names_text_only_model(monkeypatch):
    from spyre_inference.custom_ops import text_backbone

    model = nn.Sequential(nn.Linear(2, 2))
    monkeypatch.setattr(text_backbone, "_language_backbones", lambda m: [("", m)])
    assert text_backbone.non_text_parameter_names(model) == set()


def test_non_text_parameter_names_unisolatable_backbone_is_conservative(monkeypatch):
    from spyre_inference.custom_ops import text_backbone

    model = _FakeMultimodal()
    monkeypatch.setattr(text_backbone, "_language_backbones", lambda m: None)
    assert text_backbone.non_text_parameter_names(model) == {n for n, _ in model.named_parameters()}


def test_load_model_patches_attention_before_building_the_model(monkeypatch):
    """SpyreModelLoader keeps Attention buffers on the host before any module exists."""
    import spyre_inference.model_loader as loader_module
    import spyre_inference.v1.worker.model_load as model_load

    calls = []
    monkeypatch.setattr(
        model_load, "keep_attention_buffers_on_host", lambda: calls.append("patched")
    )

    def stop(**kwargs):
        calls.append("init")
        raise RuntimeError("stop")

    monkeypatch.setattr(loader_module, "initialize_model", stop)
    monkeypatch.setattr(loader_module, "_has_safetensors_checkpoint", lambda *a: True)
    loader = SpyreModelLoader(LoadConfig(load_format="spyre"))
    with pytest.raises(RuntimeError, match="stop"):
        loader.load_model(SimpleNamespace(), SimpleNamespace(dtype=torch.float16))
    assert calls == ["patched", "init"]


class _MappedModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }


def test_hf_to_vllm_name_applies_the_mapper():
    model = _MappedModel()
    assert (
        _hf_to_vllm_name(model, "model.language_model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )


def test_hf_to_vllm_name_without_a_mapper_is_a_noop():
    assert (
        _hf_to_vllm_name(nn.Linear(2, 2), "model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_mistral_format_keys_are_untouched():
    """Mistral-format keys are renamed inside the model's own load_weights, not the mapper."""
    assert _param_name_for(_MappedModel(), "tok_embeddings.weight") == "tok_embeddings.weight"


def test_param_name_for_resolves_fused_layers():
    model = _MappedModel()
    key = "model.vision_tower.transformer.layers.0.attention.q_proj.weight"
    assert (
        _param_name_for(model, key) == "vision_tower.transformer.layers.0.attention.qkv_proj.weight"
    )
    key = "model.vision_tower.transformer.layers.0.feed_forward.up_proj.weight"
    assert (
        _param_name_for(model, key)
        == "vision_tower.transformer.layers.0.feed_forward.gate_up_proj.weight"
    )


def test_hf_to_vllm_name_matches_the_real_mistral3_mapper():
    from vllm.model_executor.models.mistral3 import Mistral3ForConditionalGeneration as M

    model = SimpleNamespace(hf_to_vllm_mapper=M.hf_to_vllm_mapper)
    assert (
        _hf_to_vllm_name(model, "model.language_model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )
    # The pre-4.52 layout already matches vLLM's names and must pass through unchanged.
    legacy = "language_model.model.embed_tokens.weight"
    assert _hf_to_vllm_name(model, legacy) == legacy


def test_hf_to_vllm_name_matches_the_real_pixtral_mapper():
    from vllm.model_executor.models.pixtral import PixtralForConditionalGeneration as P

    model = SimpleNamespace(hf_to_vllm_mapper=P.hf_to_vllm_mapper)
    assert (
        _hf_to_vllm_name(model, "model.language_model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )
    assert (
        _hf_to_vllm_name(model, "model.vision_tower.transformer.layers.0.attention.q_proj.weight")
        == "vision_encoder.transformer.layers.0.attention.q_proj.weight"
    )
    legacy = "language_model.model.embed_tokens.weight"
    assert _hf_to_vllm_name(model, legacy) == legacy


def test_spyre_loader_supported_rejects_speculative_decoding():
    """The draft model reuses the target's load_config; drafters need host tensors."""
    ns = _make_ns()
    ns.speculative_config = object()
    assert not _spyre_loader_supported(ns)


def test_spyre_loader_supported_accepts_no_speculative_config():
    ns = _make_ns()
    ns.speculative_config = None
    assert _spyre_loader_supported(ns)


def _loader_and_config(tmp_path):
    loader = SpyreModelLoader(LoadConfig(load_format="spyre"))
    config = SimpleNamespace(model=str(tmp_path), revision=None, dtype=torch.float16)
    return loader, config


def test_load_model_uses_default_loader_for_bin_only_checkpoints(tmp_path, monkeypatch):
    import spyre_inference.model_loader as loader_module

    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    seen = {}

    class FakeDefaultLoader:
        def __init__(self, load_config):
            seen["load_format"] = load_config.load_format

        def load_model(self, vllm_config, model_config, prefix=""):
            seen["loaded"] = True
            return "model"

    monkeypatch.setattr(loader_module, "DefaultModelLoader", FakeDefaultLoader)
    loader, config = _loader_and_config(tmp_path)

    assert loader.load_model(SimpleNamespace(), config) == "model"
    assert seen == {"load_format": "auto", "loaded": True}
    assert loader.load_config.load_format == "spyre"  # the caller's config is untouched


def test_load_model_does_not_fall_back_when_safetensors_exist(tmp_path, monkeypatch):
    import spyre_inference.model_loader as loader_module

    (tmp_path / "model.safetensors").write_bytes(b"")

    def stop(**kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(loader_module, "initialize_model", stop)
    loader, config = _loader_and_config(tmp_path)

    with pytest.raises(RuntimeError, match="stop"):
        loader.load_model(SimpleNamespace(), config)


def _staged_linear(out_features, in_features):
    """A vLLM linear with the Spyre transposed-weight method, built without a TP group."""
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
    nn.Module.__init__(layer)
    layer.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
    layer.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
    layer.quant_method = SpyreUnquantizedLinearMethod()
    return layer


_STAGED_KEYS = ["a.weight", "norm.weight", "b.weight", "a.bias", "b.bias"]


def _staged_model(monkeypatch, window_mb):
    model = nn.Module()
    model.a, model.b = _staged_linear(4, 8), _staged_linear(4, 8)
    model.norm = nn.LayerNorm(8)
    uploads = []
    monkeypatch.setenv("SPYRE_LOAD_WINDOW_MB", str(window_mb))
    monkeypatch.setattr(
        "torch_spyre.model_utils.load_model_to_spyre",
        lambda module, num_threads: uploads.append(module),
    )
    monkeypatch.setattr("torch_spyre.model_utils._start_upload_thread", lambda device: None)
    monkeypatch.setattr(torch.spyre, "current_device", lambda: 0, raising=False)
    monkeypatch.setattr(
        "spyre_inference.custom_ops.linear._upload_transposed", lambda w: w.t().contiguous()
    )
    return model, _UploadStager(model, _STAGED_KEYS), uploads


def _load(model, name, value=None):
    """Load one checkpoint tensor the way a vLLM model does."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    param = model.get_parameter(name)
    value = torch.randn(param.shape) if value is None else value
    getattr(param, "weight_loader", default_weight_loader)(param, value)
    return value


def test_stager_uploads_a_linear_once_its_last_tensor_is_loaded(monkeypatch):
    model, stager, uploads = _staged_model(monkeypatch, window_mb=0)

    w = _load(model, "a.weight")
    _load(model, "norm.weight")
    _load(model, "b.weight")
    assert uploads == [] and model.a.weight.shape == (4, 8)  # a.bias is still to come

    _load(model, "a.bias")
    stager.close()
    assert uploads == [model.a]
    assert torch.equal(model.a.weight, w.t())
    assert model.b.weight.shape == (4, 8)


def test_stager_follows_loads_not_reads(monkeypatch):
    """A model may collect the whole iterator before loading anything."""
    model, stager, uploads = _staged_model(monkeypatch, window_mb=0)
    for key in reversed(_STAGED_KEYS):
        _load(model, key)
    stager.close()
    assert uploads == [model.b, model.a]


def test_stager_waits_while_the_window_is_full(monkeypatch):
    import threading

    model, stager, uploads = _staged_model(monkeypatch, window_mb=0)
    release, started = threading.Event(), threading.Event()

    def slow_upload(module, num_threads):
        started.set()
        assert release.wait(timeout=30)
        uploads.append(module)

    monkeypatch.setattr("torch_spyre.model_utils.load_model_to_spyre", slow_upload)
    _load(model, "a.weight")
    _load(model, "a.bias")
    assert started.wait(timeout=30) and uploads == []

    threading.Timer(0.2, release.set).start()
    _load(model, "b.weight")
    _load(model, "b.bias")  # blocks until a's upload has left the window
    assert uploads[0] is model.a
    stager.close()
    assert uploads == [model.a, model.b]


def test_an_upload_error_fails_the_load(monkeypatch):
    model, stager, _ = _staged_model(monkeypatch, window_mb=0)

    def broken(module, num_threads):
        raise RuntimeError("DMA failed")

    monkeypatch.setattr("torch_spyre.model_utils.load_model_to_spyre", broken)
    _load(model, "a.weight")
    _load(model, "a.bias")
    with pytest.raises(RuntimeError, match="DMA failed"):
        stager.close()
    stager.close()  # nothing left to raise


def test_a_failed_load_drops_upload_errors(monkeypatch):
    model, stager, _ = _staged_model(monkeypatch, window_mb=1)

    def broken(module, num_threads):
        raise RuntimeError("DMA failed")

    monkeypatch.setattr("torch_spyre.model_utils.load_model_to_spyre", broken)
    _load(model, "a.weight")
    _load(model, "a.bias")
    stager.close(failed=True)
    assert not hasattr(model.b.weight, "weight_loader")


def test_staged_linear_is_not_transposed_again_by_the_model_wide_pass(monkeypatch):
    model, stager, _ = _staged_model(monkeypatch, window_mb=0)
    _load(model, "a.weight")
    _load(model, "a.bias")
    stager.close()
    assert model.a.weight.shape == (8, 4)

    model.a.quant_method.process_weights_after_loading(model.a)
    assert model.a.weight.shape == (8, 4)
    # The skip is one-shot: a later reload processes the layer as usual.
    model.a.quant_method.process_weights_after_loading(model.a)
    assert model.a.weight.shape == (4, 8)


def test_a_tensor_arriving_after_the_upload_is_refused(monkeypatch):
    model, _, _ = _staged_model(monkeypatch, window_mb=0)
    stale = model.a.weight
    _load(model, "a.weight")
    _load(model, "a.bias")
    with pytest.raises(RuntimeError, match="after the layer was uploaded"):
        stale.weight_loader(stale, torch.zeros(4, 8))


def test_close_restores_the_loaders_of_layers_that_never_filled(monkeypatch):
    model, stager, uploads = _staged_model(monkeypatch, window_mb=0)
    _load(model, "b.weight")
    stager.close()
    assert uploads == [] and model.b.weight.shape == (4, 8)
    assert not hasattr(model.b.weight, "weight_loader")
    assert not hasattr(model.a.bias, "weight_loader")


def test_stager_leaves_host_loaded_and_distinct_weight_t_layers_alone(monkeypatch):
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    class _KeepsWeight(SpyreUnquantizedLinearMethod):
        WEIGHT_T_ATTR = "padded_weight_t"

    model = nn.Module()
    model.a, model.b = _staged_linear(4, 8), _staged_linear(4, 8)
    model.b.quant_method = _KeepsWeight()
    monkeypatch.setattr(
        "spyre_inference.model_loader._get_host_loaded_param_names", lambda model: {"a.weight"}
    )
    _UploadStager(model, ["a.weight", "b.weight"])
    assert not hasattr(model.a.weight, "weight_loader")
    assert not hasattr(model.b.weight, "weight_loader")


class _TwoLinearModel(nn.Module):
    """Two plain linears, neither of which the stager owns."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 64, bias=False, dtype=torch.float16)
        self.fallback = nn.Linear(64, 64, bias=False, dtype=torch.float16)

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            self.get_parameter(name).data.copy_(tensor)
            loaded.add(name)
        return loaded


def test_loader_reads_host_tensors_and_leaves_unstaged_layers_on_the_host(tmp_path, monkeypatch):
    """Layers the stager does not own are loaded on the host, for the runner's move."""
    w_dense = torch.randn(64, 64, dtype=torch.float16)
    w_fallback = torch.randn(64, 64, dtype=torch.float16)
    ckpt = tmp_path / "model.safetensors"
    save_file({"dense.weight": w_dense, "fallback.weight": w_fallback}, str(ckpt))
    monkeypatch.setattr(
        "spyre_inference.model_loader._resolve_checkpoint_files",
        lambda model_config, load_config: (str(tmp_path), [str(ckpt)]),
    )

    model = _TwoLinearModel()
    monkeypatch.setattr(torch.spyre, "is_initialized", lambda: True, raising=False)
    loader = SpyreModelLoader(LoadConfig(load_format="spyre"))
    loader.load_weights(model, SimpleNamespace(dtype=torch.float16))

    assert model.dense.weight.device.type == "cpu"
    assert torch.equal(model.dense.weight, w_dense)
    assert torch.equal(model.fallback.weight, w_fallback)


def test_get_all_weights_alone_installs_no_hooks(tmp_path, monkeypatch):
    """vLLM's in-place reload calls get_all_weights directly; it must stay a plain iterator."""
    ckpt = tmp_path / "model.safetensors"
    save_file({"a.weight": torch.randn(4, 8), "a.bias": torch.zeros(4)}, str(ckpt))
    monkeypatch.setattr(
        "spyre_inference.model_loader._resolve_checkpoint_files",
        lambda model_config, load_config: (str(tmp_path), [str(ckpt)]),
    )
    model = nn.Module()
    model.a = _staged_linear(4, 8)

    loader = SpyreModelLoader(LoadConfig(load_format="spyre"))
    weights = dict(loader.get_all_weights(SimpleNamespace(dtype=torch.float16), model))

    assert set(weights) == {"a.weight", "a.bias"}
    assert all(t.device.type == "cpu" for t in weights.values())
    assert not hasattr(model.a.weight, "weight_loader")
