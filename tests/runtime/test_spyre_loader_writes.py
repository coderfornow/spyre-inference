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

"""SpyreModelLoader's threaded card writes (no device needed).

The device copy is replaced by a host copy, so these cover the pool's contract: who runs a
write, ordering, the byte budget, failures and what is left behind. The device tests in
``test_spyre_model_loader.py`` check the bytes that land on the card.
"""

import contextlib
import itertools
import mmap
import os
import random
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.config import LoadConfig

import spyre_inference.custom_ops.linear as linear_module
import spyre_inference.model_loader as loader_module
from spyre_inference.model_loader import (
    SpyreModelLoader,
    _card_writes_on_threads,
    _CardWrites,
    _SwappedCheckpoint,
)


@pytest.fixture(autouse=True)
def fake_spyre(monkeypatch):
    """A started runtime with no device: the threads only need these calls, and no
    destination has a device layout."""
    monkeypatch.setattr(
        torch,
        "spyre",
        SimpleNamespace(
            current_device=lambda: 0,
            set_device=lambda idx: None,
            Stream=lambda: "stream",
            stream=lambda s: contextlib.nullcontext(),
        ),
        raising=False,
    )
    monkeypatch.setattr(linear_module, "swapped_rows", lambda t: None)


def _host_copy(dst, src):
    # Tests run in inference mode, which a new thread does not inherit.
    with torch.inference_mode():
        dst.copy_(src)


@pytest.fixture
def copies(monkeypatch):
    """Replace the device copy; returns the (thread name, source) of each one in order."""
    done = []

    def copy(dst, src):
        done.append((threading.current_thread().name, src))
        _host_copy(dst, src)

    monkeypatch.setattr(loader_module, "_copy_to_card", copy)
    return done


@contextlib.contextmanager
def _writes(threads=4):
    writes = _CardWrites(threads)
    try:
        yield writes
    finally:
        writes.shutdown()


def _load_threads_alive() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("spyre-load")]


def _gated_copy(monkeypatch, gate, fail=None):
    """A device copy whose first call waits for ``gate`` (and then raises ``fail``)."""
    order = []

    def copy(dst, src):
        first = not order
        order.append(src)
        if first:
            gate.wait(5)
            if fail is not None:
                raise fail
        _host_copy(dst, src)

    monkeypatch.setattr(loader_module, "_copy_to_card", copy)
    return order


def test_writes_run_on_load_threads(copies):
    srcs = [torch.full((4, 4), float(i)) for i in range(6)]
    dsts = [torch.empty(4, 4) for _ in srcs]
    with _writes() as writes:
        for dst, src in zip(dsts, srcs):
            writes.submit(dst, src)
        writes.wait()
    assert all(torch.equal(d, s) for d, s in zip(dsts, srcs))
    assert all(name.startswith("spyre-load") for name, _ in copies)
    assert not _load_threads_alive()


def test_one_thread_writes_on_the_caller(copies):
    dst = torch.empty(4, 4)
    with _writes(threads=1) as writes:
        writes.submit(dst, torch.ones(4, 4))
    assert [name for name, _ in copies] == [threading.current_thread().name]
    assert torch.equal(dst, torch.ones(4, 4))


def test_a_failed_write_is_raised_by_wait_and_by_the_next_submit(monkeypatch):
    def copy(dst, src):
        raise RuntimeError("device fault")

    monkeypatch.setattr(loader_module, "_copy_to_card", copy)
    with _writes() as writes:
        writes.submit(torch.empty(2), torch.ones(2))
        with pytest.raises(RuntimeError, match="device fault"):
            writes.wait()
        with pytest.raises(RuntimeError, match="device fault"):
            writes.submit(torch.empty(2), torch.ones(2))
    assert not _load_threads_alive()


def test_a_second_write_to_the_same_elements_lands_after_the_first(monkeypatch):
    gate = threading.Event()
    order = _gated_copy(monkeypatch, gate)
    dst = torch.empty(8, 4)
    first, second = torch.zeros(4, 4), torch.ones(4, 4)
    with _writes() as writes:
        writes.submit(dst[:4], first)
        threading.Timer(0.05, gate.set).start()
        writes.submit(dst[2:6], second)
        assert gate.is_set()
        writes.wait()
    assert order[0] is first and order[1] is second
    assert torch.equal(dst[2:6], second)


def test_disjoint_slices_of_one_tensor_do_not_wait(monkeypatch):
    gate = threading.Event()
    _gated_copy(monkeypatch, gate)
    dst = torch.empty(8, 4)
    with _writes() as writes:
        writes.submit(dst[:4], torch.zeros(4, 4))
        writes.submit(dst[4:], torch.ones(4, 4))
        assert not gate.is_set()
        gate.set()
        writes.wait()
    assert torch.equal(dst[4:], torch.ones(4, 4))


def test_budget_admits_one_write_above_it_and_holds_the_next(monkeypatch):
    monkeypatch.setattr(loader_module, "_MAX_OUTSTANDING_BYTES", 8)
    gate = threading.Event()
    _gated_copy(monkeypatch, gate)
    with _writes() as writes:
        writes.submit(torch.empty(64), torch.ones(64))
        threading.Timer(0.05, gate.set).start()
        writes.submit(torch.empty(64), torch.ones(64))
        assert gate.is_set()
        writes.wait()


def test_a_failure_wakes_a_submit_waiting_on_the_budget(monkeypatch):
    monkeypatch.setattr(loader_module, "_MAX_OUTSTANDING_BYTES", 8)
    gate = threading.Event()
    _gated_copy(monkeypatch, gate, fail=RuntimeError("device fault"))
    with _writes() as writes:
        writes.submit(torch.empty(64), torch.ones(64))
        threading.Timer(0.05, gate.set).start()
        with pytest.raises(RuntimeError, match="device fault"):
            writes.submit(torch.empty(64), torch.ones(64))


def test_row_slice_of_a_swapped_weight_must_sit_on_the_stick_grid(monkeypatch, copies):
    monkeypatch.setattr(linear_module, "swapped_rows", lambda t: 320)
    weight = torch.empty(320, 8, dtype=torch.float16)
    with _writes() as writes:
        with pytest.raises(ValueError, match="rows 100:220 .* not aligned"):
            writes.submit(weight[100:220], torch.ones(120, 8, dtype=torch.float16))
        with pytest.raises(ValueError, match="rows 64:164 .* not aligned"):
            writes.submit(weight[64:164], torch.ones(100, 8, dtype=torch.float16))
        writes.submit(weight[64:192], torch.ones(128, 8, dtype=torch.float16))
        writes.submit(weight[:300], torch.ones(300, 8, dtype=torch.float16))
        writes.wait()
    assert len(copies) == 2


def test_strided_source_goes_as_is_only_into_a_whole_swapped_weight(monkeypatch, copies):
    whole, other = torch.empty(8, 4), torch.empty(8, 4)
    monkeypatch.setattr(linear_module, "swapped_rows", lambda t: 32 if t is whole else None)
    shard = torch.arange(64.0).reshape(8, 8)[:, 4:]
    with _writes(threads=1) as writes:
        writes.submit(whole, shard)
        writes.submit(other, shard)
    assert [src.is_contiguous() for _, src in copies] == [False, True]
    assert torch.equal(whole, shard) and torch.equal(other, shard)


def test_copy_is_wrapped_only_inside_the_context(copies):
    assert "copy_" not in torch.Tensor.__dict__
    dst, other = torch.empty(4), torch.empty(4)
    with _card_writes_on_threads(4):
        assert "copy_" in torch.Tensor.__dict__
        # A host destination and a call from another thread are copied where they are made.
        assert dst.copy_(torch.ones(4)) is dst
        thread = threading.Thread(target=lambda: _host_copy(other, torch.ones(4)))
        thread.start()
        thread.join()
        wrapper = torch.Tensor.__dict__["copy_"]
        with _card_writes_on_threads(4):
            assert torch.Tensor.__dict__["copy_"] is wrapper
        assert torch.Tensor.__dict__["copy_"] is wrapper
    assert torch.equal(dst, torch.ones(4)) and torch.equal(other, torch.ones(4))
    assert not copies
    assert "copy_" not in torch.Tensor.__dict__
    assert not _load_threads_alive()


def test_copy_is_restored_and_threads_joined_when_the_load_raises(copies):
    with pytest.raises(RuntimeError, match="load failed"), _card_writes_on_threads(4):
        raise RuntimeError("load failed")
    assert "copy_" not in torch.Tensor.__dict__
    assert not _load_threads_alive()


@pytest.mark.parametrize("extra", [{"num_threads": 0}, {"num_threads": "4"}, {"threads": 4}])
def test_extra_config_is_validated(extra):
    SpyreModelLoader(LoadConfig(load_format="spyre", model_loader_extra_config={"num_threads": 4}))
    with pytest.raises(ValueError, match="num_threads"):
        SpyreModelLoader(LoadConfig(load_format="spyre", model_loader_extra_config=extra))


@pytest.mark.skipif(sys.platform != "linux", reason="mapped pages are released on Linux only")
def test_pages_are_released_only_inside_the_checkpoint_mapping(tmp_path, monkeypatch):
    expected = torch.randn(512, 1024)
    path = tmp_path / "model.safetensors"
    save_file({"w": expected}, str(path))
    calls = []
    real = loader_module._libc().madvise

    def madvise(start, length, advice):
        calls.append((start.value, length.value))
        return real(start, length, advice)

    monkeypatch.setattr(loader_module, "_libc", lambda: SimpleNamespace(madvise=madvise))
    writes = _CardWrites(1)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        writes.map_file(str(path))
        mapped = f.get_tensor("w")
        writes.release_pages(expected)
        assert not calls
        writes.release_pages(mapped)
        if sys.byteorder == "big":
            # safetensors hands out a byte-swapped heap copy, which must be left alone.
            assert not calls
            assert torch.equal(mapped, expected)
            return
        ((start, length),) = calls
        assert start % mmap.PAGESIZE == 0 and length % mmap.PAGESIZE == 0
        assert mapped.data_ptr() <= start and length <= mapped.numel() * 4
        # The pages fault back in from the file.
        assert torch.equal(mapped, expected)
        writes.map_file(None)
        writes.release_pages(mapped)
        assert len(calls) == 1


@pytest.mark.parametrize("threads", [1, 4])
def test_swapped_checkpoint_matches_safetensors(tmp_path, monkeypatch, threads):
    monkeypatch.setattr(loader_module, "_MIN_SWAP_CHUNK", 64)
    # Smaller than a chunk and not a divisor of one, so blocks end mid-chunk.
    monkeypatch.setattr(loader_module, "_SWAP_BLOCK", 40)
    tensors = {
        "bf16": torch.randn(33, 65).to(torch.bfloat16),
        "f16": torch.randn(7, 129).to(torch.float16),
        "f32": torch.randn(1000),
        "f64": torch.randn(131, dtype=torch.float64),
        "i16": torch.arange(-300, 301, dtype=torch.int16),
        "i32": torch.arange(-500, 500, dtype=torch.int32),
        "i64": torch.arange(-500, 500),
        "scalar": torch.tensor(3.5),
        "empty": torch.empty(0, 4),
        "u8": torch.arange(200, dtype=torch.uint8),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    with safe_open(str(path), framework="pt", device="cpu") as f:
        swapped = _SwappedCheckpoint(str(path), threads, f)
        try:
            for name, expected in tensors.items():
                got = swapped.get_tensor(name)
                assert got.dtype == expected.dtype and got.shape == expected.shape
                assert torch.equal(got, f.get_tensor(name))
        finally:
            swapped.close()
    with pytest.raises(OSError):
        os.fstat(swapped._fd)
    assert not [t.name for t in threading.enumerate() if t.name.startswith("spyre-swap")]


def test_swapped_checkpoint_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"w": torch.randn(64, 64)}, str(path))
    os.truncate(path, path.stat().st_size - 100)
    swapped = _SwappedCheckpoint(str(path), 2, None)
    try:
        with pytest.raises(OSError, match="Short read"):
            swapped.get_tensor("w")
    finally:
        swapped.close()
    assert not [t.name for t in threading.enumerate() if t.name.startswith("spyre-swap")]


def test_random_delays_and_failures_leave_nothing_behind(monkeypatch):
    monkeypatch.setattr(loader_module, "_MAX_OUTSTANDING_BYTES", 256)
    rng = random.Random(0)
    for iteration in range(200):
        fail_at = rng.randrange(40) if iteration % 4 == 0 else None
        calls = itertools.count()

        def copy(dst, src, calls=calls, fail_at=fail_at):
            call = next(calls)
            time.sleep(rng.random() * 0.0005)
            if call == fail_at:
                raise RuntimeError("device fault")
            _host_copy(dst, src)

        monkeypatch.setattr(loader_module, "_copy_to_card", copy)
        weights = [torch.empty(16, 8) for _ in range(3)]
        expected = [torch.zeros(16, 8) for _ in weights]
        failed = False
        with _writes(rng.choice((2, 4, 8))) as writes:
            try:
                for step in range(24):
                    which, row = rng.randrange(3), rng.choice((0, 4, 8, 12))
                    src = torch.full((4, 8), float(step + 1))
                    expected[which][row : row + 4] = src
                    writes.submit(weights[which][row : row + 4], src)
                writes.wait()
            except RuntimeError:
                failed = True
        assert failed == (fail_at is not None and fail_at < 24)
        if not failed:
            written = [w.nan_to_num(0) for w in weights]
            touched = [e != 0 for e in expected]
            assert all(torch.equal(w[t], e[t]) for w, e, t in zip(written, expected, touched))
        assert not _load_threads_alive()
