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

"""SpyreModelLoader — uploads weights to Spyre while the checkpoint is still loading."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)


@register_model_loader("spyre")
class SpyreModelLoader(BaseModelLoader):
    """Loads on the host a window of layers at a time, uploading each finished linear."""

    def download_model(self, model_config: ModelConfig) -> None:
        from vllm.model_executor.model_loader.weight_utils import (
            download_weights_from_hf,
        )

        download_weights_from_hf(
            model_name_or_path=model_config.model,
            cache_dir=self.load_config.download_dir,
            allow_patterns=["*.safetensors"],
            revision=model_config.revision,
        )

    def get_all_weights(
        self, model_config: ModelConfig, model: nn.Module
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, host tensor) pairs from the checkpoint shards in file order."""
        from safetensors import safe_open

        _, shard_files = _resolve_checkpoint_files(model_config, self.load_config)
        for path in shard_files:
            f = safe_open(path, framework="pt", device="cpu")
            mapped = 0
            try:
                for name in _shard_keys(f):
                    tensor = f.get_tensor(name)
                    mapped += tensor.numel() * tensor.element_size()
                    yield name, tensor
                    del tensor
                    if mapped >= _REMAP_BYTES:
                        # Reopen, so the file pages mapped so far leave this process.
                        mapped = 0
                        f.__exit__(None, None, None)
                        f = safe_open(path, framework="pt", device="cpu")
            finally:
                f.__exit__(None, None, None)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load through the model's weight loaders, uploading linears as they fill.

        The model's own weight loaders fuse and shard on the host. ``_UploadStager``
        uploads each linear layer as soon as it has been filled, so only a window of
        layers is resident on the host at a time.
        """
        if hasattr(torch, "spyre") and hasattr(torch.spyre, "is_initialized"):
            assert torch.spyre.is_initialized(), (
                "SpyreModelLoader called before torch_spyre initialization. "
                "Ensure TorchSpyreWorker.init_device() runs first."
            )
        from safetensors import safe_open

        _, shard_files = _resolve_checkpoint_files(model_config, self.load_config)
        keys = []
        for path in shard_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                keys += _shard_keys(f)
        stager = _UploadStager(model, keys)
        try:
            loaded_weights = model.load_weights(  # ty: ignore[call-non-callable,union-attr]
                self.get_all_weights(model_config, model)
            )
        except BaseException:
            stager.close(failed=True)
            raise
        stager.close()
        logger.info("SpyreModelLoader: delegated weight loading to model.load_weights()")
        _check_all_weights_loaded(model, loaded_weights)

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        # `auto` reads *.bin when a repo has no safetensors; this loader streams
        # safetensors only, so such repos load exactly as they do on the stock path.
        if not _has_safetensors_checkpoint(model_config, self.load_config):
            logger.info(
                "SpyreModelLoader: %s has no safetensors checkpoint; using DefaultModelLoader.",
                model_config.model,
            )
            stock_config = copy.copy(self.load_config)
            stock_config.load_format = "auto"
            return DefaultModelLoader(stock_config).load_model(vllm_config, model_config, prefix)

        # Before any module is built or moved; see keep_attention_buffers_on_host.
        from spyre_inference.v1.worker.model_load import keep_attention_buffers_on_host

        keep_attention_buffers_on_host()

        with set_default_torch_dtype(model_config.dtype):  # ty: ignore[invalid-argument-type]
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
            )

            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, torch.device("cpu"))
            for module in model.modules():
                module.__dict__.pop("_spyre_weights_processed", None)

        return model.eval()


def _get_host_loaded_param_names(model: nn.Module) -> set[str]:
    """Parameters whose layers are never uploaded during the load.

    - VocabParallelEmbedding weights are placed by the module's own ``_apply`` in
      ``model.to(spyre)``.
    - Everything outside a multimodal model's language backbone (vision tower,
      projector). Their ``process_weights_after_loading`` hooks run eager ops on the
      weight (e.g. Pixtral pads the patch embedding) and expect a host tensor, as under
      the stock loader.

    Pooling heads are not listed: pooling runners are excluded at the platform gate
    (torch-spyre#2971).
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    from spyre_inference.custom_ops.text_backbone import non_text_parameter_names

    names = non_text_parameter_names(model)
    for module_name, module in model.named_modules():
        if isinstance(module, VocabParallelEmbedding):
            for param_name in module._parameters:
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


def _hf_to_vllm_name(model: nn.Module, key: str) -> str:
    """A checkpoint key after the model's ``hf_to_vllm_mapper`` (a no-op without one).

    ``get_all_weights`` sees raw checkpoint keys, but the mapper runs inside
    ``model.load_weights``, after the loader has already chosen host or device per key.
    A key the mapper drops is returned unchanged.
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return key
    mapped = mapper.apply_list([key])
    return mapped[0] if mapped else key


def _param_name_for(model: nn.Module, key: str) -> str:
    """The vLLM parameter name a checkpoint key loads into: mapper first, then the
    fused-layer name (``q_proj`` -> ``qkv_proj``) from ``packed_modules_mapping``."""
    name = _hf_to_vllm_name(model, key)
    for packed, shards in getattr(model, "packed_modules_mapping", {}).items():
        for shard in shards:
            if f".{shard}." in name:
                return name.replace(f".{shard}.", f".{packed}.", 1)
    return name


def _check_all_weights_loaded(model: nn.Module, loaded_weights: set[str] | None) -> None:
    """Fail if a parameter received no checkpoint tensor, as DefaultModelLoader does.

    ``loaded_weights`` is what ``model.load_weights()`` returned; a model that does not
    track it (None) cannot be checked.
    """
    if loaded_weights is None:
        logger.warning(
            "SpyreModelLoader: %s.load_weights() did not report which weights it loaded; "
            "skipping the uninitialized-parameter check (as the stock loader does). A "
            "parameter missing from the checkpoint would keep its uninitialized placeholder.",
            type(model).__name__,
        )
        return
    weights_not_loaded = {name for name, _ in model.named_parameters()} - loaded_weights
    if weights_not_loaded:
        raise ValueError(
            f"Following weights were not initialized from checkpoint: {sorted(weights_not_loaded)}"
        )


_REMAP_BYTES = 256 << 20


def _shard_keys(f) -> list[str]:
    """A shard's keys in file order where safetensors can tell, so reads are sequential."""
    return list(getattr(f, "offset_keys", f.keys)())


def _get_submodule(model: nn.Module, path: str) -> nn.Module | None:
    try:
        return model.get_submodule(path)
    except AttributeError:
        return None


class _UploadStager:
    """Uploads each transposed-weight linear once the checkpoint has filled it.

    Every checkpoint key that loads into such a linear is counted up front, and the
    layer's parameters get a ``weight_loader`` that counts the loads down. At zero the
    weight is transposed on the host, exactly as the stock path does after loading, and
    the layer is handed to an upload thread while loading carries on. Loading waits
    whenever more than ``SPYRE_LOAD_WINDOW_MB`` of weights are queued or in flight, which
    bounds the host copy. Counting loads rather than keys read keeps this correct for a
    model that collects the whole iterator before loading.

    A layer whose count never reaches zero, and everything that is not such a linear,
    stays on the host for vLLM's post-load pass and the runner's ``move_model_to_spyre``.
    """

    def __init__(self, model: nn.Module, keys: list[str]) -> None:
        from vllm.model_executor.layers.linear import LinearBase

        from spyre_inference.custom_ops.linear import SpyreTransposedWeightMethod

        self.window = int(os.environ.get("SPYRE_LOAD_WINDOW_MB", "1024")) << 20
        self._left: dict[int, int] = {}
        self._hooks: dict[int, list[tuple[nn.Parameter, bool, object]]] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._in_flight: dict[Future, int] = {}
        host_names = _get_host_loaded_param_names(model)
        for key in keys:
            name = _param_name_for(model, key)
            module_name, _, param_name = name.rpartition(".")
            module = _get_submodule(model, module_name)
            method = getattr(module, "quant_method", None)
            if not (
                isinstance(module, LinearBase)
                and isinstance(method, SpyreTransposedWeightMethod)
                and method.WEIGHT_T_ATTR == "weight"
                and not any(f"{module_name}.{n}" in host_names for n in module._parameters)
            ):
                continue
            param = module._parameters.get(param_name)
            if param is None:
                continue
            if id(module) not in self._left:
                self._left[id(module)] = 0
                self._hooks[id(module)] = []
            self._left[id(module)] += 1
            if not any(p is param for p, _, _ in self._hooks[id(module)]):
                self._hook(module, param)

    def _hook(self, module: nn.Module, param: nn.Parameter) -> None:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        had_loader = hasattr(param, "weight_loader")
        load = getattr(param, "weight_loader", default_weight_loader)

        def weight_loader(p, *args, **kwargs):
            load(p, *args, **kwargs)
            self._loaded(module)

        param.weight_loader = weight_loader  # ty: ignore[unresolved-attribute]
        self._hooks[id(module)].append((param, had_loader, load))

    def _unhook(self, module: nn.Module, stale: nn.Parameter | None = None) -> None:
        for param, had_loader, load in self._hooks.pop(id(module), ()):
            if param is stale:
                param.weight_loader = _refuse_late_load  # ty: ignore[unresolved-attribute]
            elif had_loader:
                param.weight_loader = load  # ty: ignore[unresolved-attribute]
            else:
                del param.weight_loader  # ty: ignore[unresolved-attribute]
        self._left.pop(id(module), None)

    def _loaded(self, module: nn.Module) -> None:
        self._left[id(module)] -= 1
        if self._left[id(module)]:
            return
        # build_weight_t replaces `weight`; a later load into the old one would be lost.
        self._unhook(module, stale=module._parameters["weight"])
        if self._pool is None:
            from torch_spyre.model_utils import _start_upload_thread, _upload_thread_count

            # The same threads and per-thread streams as load_model_to_spyre's own pool.
            self._pool = ThreadPoolExecutor(
                max_workers=_upload_thread_count(None),
                thread_name_prefix="spyre-load",
                initializer=_start_upload_thread,
                initargs=(torch.spyre.current_device(),),  # ty: ignore[unresolved-attribute]
            )
        size = sum(p.numel() * p.element_size() for p in module.parameters())
        self._wait(self.window - size)
        self._in_flight[self._pool.submit(_process_and_upload, module)] = size

    def _wait(self, limit: int) -> None:
        """Block until at most ``limit`` bytes are in flight; raise an upload's error."""
        while self._in_flight:
            done = [f for f in self._in_flight if f.done()]
            if not done and sum(self._in_flight.values()) > limit:
                done = wait(self._in_flight, return_when=FIRST_COMPLETED).done
            if not done:
                return
            for future in done:
                del self._in_flight[future]
                future.result()

    def close(self, failed: bool = False) -> None:
        """Finish the uploads and drop the hooks of the layers that never filled.

        After a failed load the queued uploads are cancelled and their errors dropped,
        so the load's own error is the one raised.
        """
        try:
            if not failed:
                self._wait(-1)
        finally:
            self._in_flight.clear()
            if self._pool is not None:
                self._pool.shutdown(cancel_futures=True)
                self._pool = None
            for hooks in list(self._hooks.values()):
                for param, had_loader, load in hooks:
                    if had_loader:
                        param.weight_loader = load  # ty: ignore[unresolved-attribute]
                    else:
                        del param.weight_loader  # ty: ignore[unresolved-attribute]
            self._hooks.clear()
            self._left.clear()


def _process_and_upload(module: nn.Module) -> None:
    from torch_spyre.model_utils import load_model_to_spyre

    # Read once each by SpyreTransposedWeightMethod: the weight goes to the device as it
    # is transposed, and vLLM's pass over the whole model does not process it again.
    module._spyre_upload_weight_t = True  # ty: ignore[unresolved-attribute]
    module.quant_method.process_weights_after_loading(module)  # ty: ignore[call-non-callable,unresolved-attribute]
    module._spyre_weights_processed = True  # ty: ignore[unresolved-attribute]
    if hasattr(module, "update_param_tp_status"):
        module.update_param_tp_status()  # ty: ignore[call-non-callable]
    load_model_to_spyre(module, num_threads=1)


def _refuse_late_load(param, *args, **kwargs):
    raise RuntimeError(
        "SpyreModelLoader: a checkpoint tensor arrived for a linear layer after the layer "
        "was uploaded; use --load-format safetensors for this model."
    )


def _has_safetensors_checkpoint(model_config: ModelConfig, load_config) -> bool:
    """Whether the checkpoint has safetensors shards (fetched first when remote)."""
    try:
        _resolve_checkpoint_files(model_config, load_config)
    except FileNotFoundError:
        return False
    return True


def _resolve_checkpoint_files(
    model_config: ModelConfig,
    load_config,
) -> tuple[str, list[str]]:
    """Resolve and filter shard file paths for the checkpoint."""
    import glob as glob_mod

    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
    from vllm.model_executor.model_loader.weight_utils import (
        download_safetensors_index_file_from_hf,
        download_weights_from_hf,
        filter_duplicate_safetensors_files,
    )

    model_path = model_config.model
    revision = model_config.revision
    is_local = os.path.isdir(model_path)

    if not is_local:
        model_path = download_weights_from_hf(
            model_name_or_path=model_path,
            cache_dir=load_config.download_dir,
            allow_patterns=["*.safetensors"],
            revision=revision,
            ignore_patterns=load_config.ignore_patterns,
        )

    shard_files = glob_mod.glob(os.path.join(model_path, "*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(
            f"No safetensors checkpoint shards found in {model_path}. SpyreModelLoader "
            "reads safetensors only; for .bin checkpoints pass the necessary "
            "--load-format pt."
        )
    if not is_local and len(shard_files) > 1:
        download_safetensors_index_file_from_hf(
            model_config.model,
            SAFE_WEIGHTS_INDEX_NAME,
            cache_dir=load_config.download_dir,
            revision=revision,
        )
    shard_files = sorted(
        filter_duplicate_safetensors_files(shard_files, model_path, SAFE_WEIGHTS_INDEX_NAME)
    )

    return model_path, shard_files
