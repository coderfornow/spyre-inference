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

"""Unit tests for SpyreModelLoader and direct-to-device weight loading."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.config import LoadConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader

import spyre_inference.platform as plat_mod
from spyre_inference.custom_ops.linear import (
    SpyreMergedColumnParallelLinear,
    SpyreQKVParallelLinear,
    SpyreRowParallelLinear,
    SpyreTransposedWeightMethod,
    swapped_rows,
    swapped_weight_layout,
)
from spyre_inference.model_loader import (
    SpyreModelLoader,
    _card_writes_on_threads,
    _check_all_weights_loaded,
    _get_host_loaded_param_names,
    _placeholder,
    _resolve_checkpoint_files,
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


@pytest.fixture
def on_power(monkeypatch):
    """Pretend the host is POWER, where `auto` resolves to the spyre loader.

    Set before the VllmConfig is built: construction already runs the platform hook.
    """
    monkeypatch.setattr(plat_mod, "_host_machine", lambda: "ppc64le")


@pytest.mark.parametrize("machine", ["ppc64le", "ppc64"])
def test_platform_defaults_load_format_to_spyre_on_power(monkeypatch, machine):
    """Positive control: a plain TP=1 text-generation config flips auto → spyre on POWER."""
    monkeypatch.setattr(plat_mod, "_host_machine", lambda: machine)
    vllm_config = _make_vllm_config()
    assert vllm_config.load_config.load_format == "spyre"


@pytest.mark.parametrize("machine", ["x86_64", "aarch64", "s390x"])
def test_platform_keeps_auto_load_format_off_power(monkeypatch, machine):
    """spyre is opt-in on every other architecture: a supported config stays on 'auto'."""
    monkeypatch.setattr(plat_mod, "_host_machine", lambda: machine)
    vllm_config = _make_vllm_config()
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "auto"


def test_platform_defaults_load_format_to_spyre_tp2(on_power):
    """TP>1 is supported: auto resolves to spyre (linear weights shard on read)."""
    vllm_config = _make_vllm_config(tp_size=2)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "spyre"


def test_platform_defaults_load_format_to_spyre_multimodal(on_power):
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
def test_platform_load_format_stays_auto(kwargs, on_power):
    """Even on POWER, load_format stays 'auto' for configs SpyreModelLoader doesn't support."""
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
    """TP > 1 and multimodal models are supported by the direct-to-device loader."""
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


def test_platform_load_format_stays_auto_transformers_backend(on_power):
    """load_format stays 'auto' for the transformers backend."""
    assert _make_vllm_config(transformers_backend=True).load_config.load_format == "auto"


def test_platform_load_format_stays_auto_moe(on_power):
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


def test_gate_sees_padding_applied_by_the_hook(monkeypatch, on_power):
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


class _TwoLinearModel(nn.Module):
    """A normal linear layer plus one whose weight must stay on the host."""

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


def _exact(*shape):
    """Values the card's 16-bit format holds exactly, so a read-back compares with ``equal``."""
    return torch.randn(*shape).to(torch.bfloat16).to(torch.float16)


def _weight_t(layer, parts, *, on_card: bool):
    """``layer``'s stored ``W^T`` after loading ``parts`` ((shard id or None, tensor) pairs),
    through the stock host path or through SpyreModelLoader's placeholder and card writes."""
    if on_card:
        layer._parameters["weight"] = _placeholder(layer, "weight", layer.weight)
    with _card_writes_on_threads(4 if on_card else 1):
        for shard_id, tensor in parts:
            args = (tensor,) if shard_id is None else (tensor, shard_id)
            layer.weight_loader(layer.weight, *args)
    layer.quant_method.process_weights_after_loading(layer)
    return layer.weight


_FUSED_CASES = [
    pytest.param(
        lambda: SpyreQKVParallelLinear(256, 64, 4, 2, bias=False, params_dtype=torch.float16),
        [("q", (256, 256)), ("k", (128, 256)), ("v", (128, 256))],
        True,
        id="qkv",
    ),
    pytest.param(
        lambda: SpyreMergedColumnParallelLinear(
            256, [512, 512], bias=False, params_dtype=torch.float16
        ),
        [(0, (512, 256)), (1, (512, 256))],
        True,
        id="gate-up",
    ),
    pytest.param(
        lambda: SpyreMergedColumnParallelLinear(
            256, [100, 220], bias=False, params_dtype=torch.float16
        ),
        [(0, (100, 256)), (1, (220, 256))],
        False,
        id="gate-up-off-the-stick-grid",
    ),
    pytest.param(
        lambda: SpyreRowParallelLinear(256, 1000, bias=False, params_dtype=torch.float16),
        [(None, (1000, 256))],
        True,
        id="row-parallel",
    ),
]


@pytest.mark.spyre_device
@pytest.mark.parametrize(("make", "shards", "swapped"), _FUSED_CASES)
def test_card_written_weight_equals_the_stock_transpose(make, shards, swapped, tp_group):
    if not spyre_available():
        pytest.skip("Spyre device not available")
    parts = [(shard_id, _exact(*shape)) for shard_id, shape in shards]
    stock = _weight_t(make(), parts, on_card=False)

    layer = make()
    assert (swapped_rows(_placeholder(layer, "weight", layer.weight)) is not None) == swapped
    direct = _weight_t(layer, parts, on_card=True)

    assert direct.device.type == "spyre"
    assert swapped_rows(direct) is None
    assert torch.equal(direct.cpu(), stock)


@pytest.mark.spyre_device
def test_tp_column_shard_is_written_without_a_host_copy(tp_group):
    """A row-parallel rank's share is a strided view of the mapped checkpoint tensor."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    full = _exact(1024, 512)
    shard = full[:, 256:]
    assert not shard.is_contiguous()
    layer = SpyreRowParallelLinear(256, 1024, bias=False, params_dtype=torch.float16)

    assert torch.equal(_weight_t(layer, [(None, shard)], on_card=True).cpu(), shard.t())


@pytest.mark.spyre_device
def test_row_slice_off_the_stick_grid_is_rejected_on_the_card():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from torch_spyre.model_utils import spyre_empty_with_layout

    layout = swapped_weight_layout((320, 256), torch.float16)
    weight = spyre_empty_with_layout(
        torch.Size((320, 256)), (256, 1), torch.float16, layout, device=None
    )
    with (
        _card_writes_on_threads(2),
        pytest.raises(ValueError, match="rows 100:220 .* not aligned"),
    ):
        weight.narrow(0, 100, 120).copy_(_exact(120, 256))


@pytest.mark.spyre_device
def test_row_padded_weight_keeps_the_default_layout():
    """An lm-head pads its rows on the host, so its placeholder is not stick-swapped."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    class Padded(SpyreTransposedWeightMethod):
        ROW_ALIGN = 64

    module = SimpleNamespace(quant_method=Padded())
    weight = torch.empty(128, 256, dtype=torch.float16)
    assert swapped_rows(_placeholder(module, "weight", weight)) is None


def test_consolidated_checkpoint_is_read_alone(tmp_path):
    for name in ("consolidated.safetensors", "model-00001-of-00002.safetensors"):
        (tmp_path / name).write_bytes(b"")
    config = SimpleNamespace(model=str(tmp_path), revision=None)

    _, files = _resolve_checkpoint_files(config, LoadConfig(load_format="spyre"))

    assert files == [str(tmp_path / "consolidated.safetensors")]


@pytest.mark.spyre_device
@pytest.mark.parametrize("num_threads", [1, 4])
def test_loader_writes_a_bf16_checkpoint_split_over_two_files(tmp_path, monkeypatch, num_threads):
    if not spyre_available():
        pytest.skip("Spyre device not available")
    w_dense = torch.randn(128, 64, dtype=torch.bfloat16)
    w_fallback = torch.randn(64, 64, dtype=torch.bfloat16)
    files = [
        tmp_path / "model-00001-of-00002.safetensors",
        tmp_path / "model-00002-of-00002.safetensors",
    ]
    save_file({"dense.weight": w_dense}, str(files[0]))
    save_file({"fallback.weight": w_fallback}, str(files[1]))
    monkeypatch.setattr(
        "spyre_inference.model_loader._resolve_checkpoint_files",
        lambda model_config, load_config: (str(tmp_path), [str(f) for f in files]),
    )

    model = _TwoLinearModel()
    model.dense.weight = nn.Parameter(
        torch.empty(128, 64, dtype=torch.float16, device="spyre"), requires_grad=False
    )
    loader = SpyreModelLoader(
        LoadConfig(load_format="spyre", model_loader_extra_config={"num_threads": num_threads})
    )
    loader.load_weights(model, SimpleNamespace(dtype=torch.float16))

    assert torch.equal(model.dense.weight.cpu(), w_dense.to(torch.float16))
    assert model.dense.weight.device.type == "spyre"
    assert model.fallback.weight.device.type == "cpu"
    assert torch.equal(model.fallback.weight, w_fallback.to(torch.float16))
