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

"""SpyreModelLoader — direct-to-device weight loader for IBM Spyre."""

from __future__ import annotations

import contextvars
import copy
import ctypes
import functools
import json
import mmap
import os
import sys
import threading
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
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

from spyre_inference.custom_ops.linear import (
    SpyreTransposedWeightMethod,
    swapped_weight_layout,
    whole_swapped_weight,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

_DEFAULT_LOAD_THREADS = 8
_MAX_LOAD_THREADS = 16
# Bytes of checkpoint tensors queued or being written at once. The device runtime converts
# each write on the host first, so this also bounds that staging.
_MAX_OUTSTANDING_BYTES = 2 << 30

_load_thread = threading.local()


def _start_load_thread(device_index: int) -> None:
    torch.spyre.set_device(device_index)
    _load_thread.stream = torch.spyre.Stream()


def _copy_to_card(dst: torch.Tensor, src: torch.Tensor) -> None:
    from torch_spyre._C import copy_tensor

    copy_tensor(src, dst, non_blocking=False)


def _span(t: torch.Tensor) -> int:
    """Elements from the first to the last one `t` addresses in its storage."""
    return sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1 if t.numel() else 0


class _CardWrites:
    """The host-to-card weight writes of one load, on threads with a device stream each.

    `submit` returns before the data is on the card; `wait` returns once it all is."""

    def __init__(self, threads: int) -> None:
        self._pool = (
            ThreadPoolExecutor(
                max_workers=threads,
                thread_name_prefix="spyre-load",
                initializer=_start_load_thread,
                initargs=(torch.spyre.current_device(),),
            )
            if threads > 1
            else None
        )
        self._cond = threading.Condition()
        # (storage, first element, last element + 1, source bytes) of each unfinished write.
        self._pending: list[tuple[int, int, int, int]] = []
        self._bytes = 0
        self._error: BaseException | None = None
        # [start, end) addresses of the open checkpoint file's mappings.
        self._mapped: list[tuple[int, int]] = []

    def submit(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if not whole_swapped_weight(dst) and not src.is_contiguous():
            mapped = src
            src = src.contiguous()
            self.release_pages(mapped)
        if self._pool is None:
            _copy_to_card(dst, src)
            self.release_pages(src)
            return

        first = int(dst.storage_offset())
        nbytes = src.numel() * src.element_size()
        entry = (dst.untyped_storage()._cdata, first, first + _span(dst), nbytes)
        with self._cond:
            # A second write to the same elements lands after the first. One write is
            # always admitted, so a tensor above the budget cannot hang.
            self._cond.wait_for(
                lambda: (
                    self._error is not None
                    or (
                        not any(
                            s == entry[0] and lo < entry[2] and first < hi
                            for s, lo, hi, _ in self._pending
                        )
                        and (not self._pending or self._bytes + nbytes <= _MAX_OUTSTANDING_BYTES)
                    )
                )
            )
            if self._error is not None:
                raise self._error
            self._pending.append(entry)
            self._bytes += nbytes
        self._pool.submit(self._write, dst, src, entry)

    def _write(
        self, dst: torch.Tensor, src: torch.Tensor, entry: tuple[int, int, int, int]
    ) -> None:
        error = None
        try:
            with torch.spyre.stream(_load_thread.stream):
                _copy_to_card(dst, src)
            self.release_pages(src)
        except BaseException as e:
            error = e
        with self._cond:
            self._error = self._error or error
            self._pending.remove(entry)
            self._bytes -= entry[3]
            self._cond.notify_all()

    def wait(self) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._error is not None or not self._pending)
            if self._error is not None:
                raise self._error

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)

    def map_file(self, path: str | None) -> None:
        """Note where checkpoint file `path` is mapped; None forgets the last file."""
        self._mapped = []
        if path is None or sys.platform != "linux":
            return
        real = os.path.realpath(path)
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].rstrip("\n") == real:
                    start, end = fields[0].split("-")
                    self._mapped.append((int(start, 16), int(end, 16)))

    def release_pages(self, src: torch.Tensor) -> None:
        """Drop the whole pages `src` covers from this process once they are on the card.

        They stay in the page cache and fault back in if read again, as the file holds
        them: the mapping is private, so an in-place edit of a source tensor is lost.
        Only a range inside the checkpoint file's mapping is touched: the same call zeroes
        heap memory, which is what a big-endian host's byte-swapped tensors live in."""
        if not self._mapped or not src.numel():
            return
        page = mmap.PAGESIZE
        start = -(-src.data_ptr() // page) * page
        end = (src.data_ptr() + _span(src) * src.element_size()) // page * page
        if end > start and any(lo <= start and end <= hi for lo, hi in self._mapped):
            _libc().madvise(
                ctypes.c_void_p(start), ctypes.c_size_t(end - start), mmap.MADV_DONTNEED
            )


@functools.cache
def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None)


_SWAPPED_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
}
_SWAPPED_INTS = {2: torch.int16, 4: torch.int32, 8: torch.int64}
_MIN_SWAP_CHUNK = 1 << 20


class _SwappedCheckpoint:
    """A safetensors file's tensors in host byte order, converted on threads.

    For a big-endian host: safetensors swaps each tensor on the calling thread under the
    GIL, which is most of a load there. The heap copy is the one it makes too."""

    def __init__(self, path: str, threads: int) -> None:
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            self._header = json.loads(f.read(header_len))
            self._map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._base = 8 + header_len
        self._threads = threads
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="spyre-swap")

    def get_tensor(self, name: str) -> torch.Tensor | None:
        """None for a dtype with nothing to swap; safetensors reads those."""
        info = self._header[name]
        dtype = _SWAPPED_DTYPES.get(info["dtype"])
        start, end = (self._base + offset for offset in info["data_offsets"])
        if dtype is None or end == start:
            return None
        size = dtype.itemsize
        src = np.frombuffer(self._map, f"<i{size}", (end - start) // size, start)
        out = torch.empty(src.size, dtype=_SWAPPED_INTS[size])
        dst = out.numpy()
        step = max(-(-src.size // self._threads), _MIN_SWAP_CHUNK)
        list(
            self._pool.map(
                lambda i: np.copyto(dst[i : i + step], src[i : i + step]),
                range(0, src.size, step),
            )
        )
        if sys.platform == "linux":
            page = mmap.PAGESIZE
            first = -(-start // page) * page
            if end // page * page > first:
                self._map.madvise(mmap.MADV_DONTNEED, first, end // page * page - first)
        return out.view(dtype).reshape(info["shape"])

    def close(self) -> None:
        self._pool.shutdown()
        self._map.close()


_active_writes: contextvars.ContextVar[_CardWrites | None] = contextvars.ContextVar(
    "spyre_card_writes", default=None
)
_copy_patch = threading.Lock()


@contextmanager
def _card_writes_on_threads(threads: int) -> Iterator[None]:
    """Run the host-to-card `copy_` calls of the weight loaders inside on `threads` threads.

    vLLM's weight loaders end in `param.data[...].copy_(loaded_weight)`, so `Tensor.copy_`
    is wrapped while the context is open, for calls from the entering thread only. A load
    that starts while another holds the wrapper is not wrapped."""
    if not _copy_patch.acquire(blocking=False):
        logger.warning("SpyreModelLoader: another load is writing weights; loading serially.")
        yield
        return
    writes = _CardWrites(threads)
    owner = threading.current_thread()
    original = torch.Tensor.copy_
    shadowed = torch.Tensor.__dict__.get("copy_")

    def copy_(dst, src, *args, **kwargs):
        if (
            threading.current_thread() is not owner
            or not isinstance(src, torch.Tensor)
            or src.device.type != "cpu"
        ):
            return original(dst, src, *args, **kwargs)
        if dst.device.type != "spyre":
            original(dst, src, *args, **kwargs)
            writes.release_pages(src)
            return dst
        writes.submit(dst, src)
        return dst

    token = _active_writes.set(writes)
    torch.Tensor.copy_ = copy_  # ty: ignore[invalid-assignment]
    try:
        yield
        writes.wait()
    finally:
        writes.shutdown()
        if shadowed is None:
            del torch.Tensor.copy_
        else:
            torch.Tensor.copy_ = shadowed
        _active_writes.reset(token)
        _copy_patch.release()


@register_model_loader("spyre")
class SpyreModelLoader(BaseModelLoader):
    """Writes weights from safetensors straight to Spyre device memory."""

    def __init__(self, load_config) -> None:
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict) or set(extra_config) - {"num_threads"}:
            raise ValueError(
                f"model_loader_extra_config for load format spyre takes only num_threads, "
                f"got {extra_config!r}"
            )
        num_threads = extra_config.get("num_threads")
        if num_threads is not None and not (isinstance(num_threads, int) and num_threads > 0):
            raise ValueError(f"num_threads must be a positive integer, got {num_threads!r}")
        self._num_threads = num_threads

    def _threads(self) -> int:
        threads = self._num_threads or min(_DEFAULT_LOAD_THREADS, torch.get_num_threads())
        return min(threads, _MAX_LOAD_THREADS)

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
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield (name, memory-mapped tensor) pairs in physical disk order. Inside
        `_card_writes_on_threads`, each file's writes finish before the next file opens."""
        from safetensors import safe_open

        _, shard_files = _resolve_checkpoint_files(model_config, self.load_config)
        writes = _active_writes.get()
        for index, shard_path in enumerate(shard_files, 1):
            swapped = (
                _SwappedCheckpoint(shard_path, self._threads()) if sys.byteorder == "big" else None
            )
            try:
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    if writes is not None:
                        writes.map_file(shard_path)
                    for name in f.offset_keys():
                        tensor = swapped.get_tensor(name) if swapped is not None else None
                        yield name, f.get_tensor(name) if tensor is None else tensor
                    if writes is not None:
                        writes.wait()
                        writes.map_file(None)
            finally:
                if swapped is not None:
                    swapped.close()
            logger.info(
                "SpyreModelLoader: %d of %d checkpoint files written", index, len(shard_files)
            )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Stream checkpoint weights directly to Spyre device and CPU fallback modules."""
        if hasattr(torch, "spyre") and hasattr(torch.spyre, "is_initialized"):
            assert torch.spyre.is_initialized(), (
                "SpyreModelLoader called before torch_spyre initialization. "
                "Ensure TorchSpyreWorker.init_device() runs first."
            )

        weights = self.get_all_weights(model_config, model)
        try:
            with _card_writes_on_threads(self._threads()):
                loaded_weights = model.load_weights(weights)  # ty: ignore[call-non-callable,union-attr]
        finally:
            weights.close()
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

            # Swap non-host CPU parameters for uninitialized Spyre tensors before loading.
            host_names = _get_host_loaded_param_names(model)
            for module_name, module in model.named_modules():
                for param_name, param in list(module._parameters.items()):
                    if param is None:
                        continue
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    if full_name in host_names:
                        continue
                    module._parameters[param_name] = _placeholder(module, param_name, param)

            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, torch.device("cpu"))

        return model.eval()


def _placeholder(module: nn.Module, param_name: str, param: torch.Tensor) -> torch.Tensor:
    """An uninitialized device parameter in place of `param`."""
    method = getattr(module, "quant_method", None)
    if (
        param_name == "weight"
        and isinstance(method, SpyreTransposedWeightMethod)
        and method.takes_swapped_weight(module, param)
    ):
        from torch_spyre.model_utils import spyre_empty_with_layout

        layout = swapped_weight_layout(param.shape, param.dtype)
        data = spyre_empty_with_layout(
            param.size(), param.stride(), param.dtype, layout, device=None
        )
    else:
        data = torch.empty(param.shape, dtype=param.dtype, device="spyre")
    placeholder = torch.Tensor._make_subclass(type(param), data, param.requires_grad)
    placeholder.__dict__.update(param.__dict__)
    return placeholder


def _get_host_loaded_param_names(model: nn.Module) -> set[str]:
    """Parameters that must stay on CPU rather than be swapped for an empty Spyre placeholder.

    - VocabParallelEmbedding weights are placed by the module's own ``_apply`` after
      ``model.to(spyre)``; a direct Spyre-to-Spyre copy of a vocab-sized table exceeds
      the per-core limit, so they must arrive from the host.
    - Everything outside a multimodal model's language backbone (vision tower,
      projector). Their ``process_weights_after_loading`` hooks run eager ops on the
      weight (e.g. Pixtral pads the patch embedding) and expect a host tensor, as under
      the stock loader. They are small, so keeping them on the host costs little.

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


def _check_all_weights_loaded(model: nn.Module, loaded_weights: set[str] | None) -> None:
    """Fail if a parameter received no checkpoint tensor, as DefaultModelLoader does.

    ``load_model`` swaps every non-host parameter for an uninitialized
    ``torch.empty(..., device="spyre")`` placeholder, so a parameter the checkpoint never
    fills (renamed key, WeightsMapper miss, skipped prefix) would otherwise serve garbage
    silently. ``loaded_weights`` is what ``model.load_weights()`` returned; a model that
    does not track it (None) cannot be checked.
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
    """Resolve the checkpoint files the way ``DefaultModelLoader`` does for ``auto``.

    A repo with ``consolidated*.safetensors`` is read from those alone: its HF-named shards hold
    the same weights again, and vLLM's Mistral models do not map every HF name.
    """
    import glob as glob_mod

    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
    from vllm.model_executor.model_loader.weight_utils import (
        download_safetensors_index_file_from_hf,
        download_weights_from_hf,
        filter_duplicate_safetensors_files,
    )
    from vllm.transformers_utils.repo_utils import list_filtered_repo_files

    model_path = model_config.model
    revision = model_config.revision
    is_local = os.path.isdir(model_path)
    pattern, index_file = "*.safetensors", SAFE_WEIGHTS_INDEX_NAME
    if list_filtered_repo_files(model_path, ["consolidated*.safetensors"], revision):
        pattern, index_file = "consolidated*.safetensors", "consolidated.safetensors.index.json"

    if not is_local:
        model_path = download_weights_from_hf(
            model_name_or_path=model_path,
            cache_dir=load_config.download_dir,
            allow_patterns=[pattern],
            revision=revision,
            ignore_patterns=load_config.ignore_patterns,
        )

    shard_files = glob_mod.glob(os.path.join(model_path, pattern))
    if not shard_files:
        raise FileNotFoundError(
            f"No safetensors checkpoint shards found in {model_path}. SpyreModelLoader "
            "reads safetensors only; for .bin checkpoints pass the necessary "
            "--load-format pt."
        )
    if not is_local and len(shard_files) > 1:
        download_safetensors_index_file_from_hf(
            model_config.model,
            index_file,
            cache_dir=load_config.download_dir,
            revision=revision,
        )
    shard_files = sorted(filter_duplicate_safetensors_files(shard_files, model_path, index_file))

    return model_path, shard_files
