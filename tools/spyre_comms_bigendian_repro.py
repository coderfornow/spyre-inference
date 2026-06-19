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
"""Minimal spyreccl collective reproducer (TP=2) — torch + torch_spyre only.

Share with the spyre-comms team. Runs each libspyre_comms collective on a real
spyreccl group with KNOWN, rank-dependent values and prints, per collective:

  PASS            received the correct value
  FAIL/UNTOUCHED  receive buffer still holds the pre-filled sentinel -> the
                  transfer's completion never landed (data-plane stall)
  FAIL/WRONG      wrong value delivered (e.g. byte-swapped on big-endian s390x)
  UNIMPLEMENTED   the native collective throws (not implemented yet)

It does NOT abort on the first failure, so the summary makes it obvious which
collective (reduce vs all_gather vs the broadcast/send-recv primitives the
all_reduce fallback rides on) is the culprit.

Context: s390x is big-endian, the AIU is little-endian. The OOB rendezvous
rank-exchange fix lives on spyre-comms branch s390x/bigendian-rank-fix; WITHOUT
it init_process_group fails before any collective runs. Anything that FAILs
below is a separate DATA-plane big-endian defect (collective serialization or
the flex HDMA completion handshake), not covered by the rank-exchange fix.

Run (inside the spyre-inference s390x container, on a 2-PF pod):

    source /etc/profile.d/ibm-aiu-setup.sh        # sets AIU_WORLD_SIZE=2 etc.
    source /home/senuser/.venv/bin/activate
    torchrun --nproc-per-node=2 spyre_comms_bigendian_repro.py

Container needs the TP=2 resource flags:
    --pids-limit=-1 --ulimit nproc=65535:65535 -e OMP_NUM_THREADS=8

Byte-level endianness demo only (no hardware / torchrun / devices needed -
runs anywhere, shows WHY rank 1 is read as 0 on big-endian):

    python3 spyre_comms_bigendian_repro.py --demo
"""

import os
import struct
import sys

# Control when libspyre_comms loads (it reads RANK/WORLD_SIZE/LOCAL_RANK/
# LOCAL_WORLD_SIZE at dlopen). torchrun has already exported those; just make
# sure LOCAL_WORLD_SIZE is present, then autoload manually after import.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.setdefault("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))

# torch is imported lazily inside main() (after the --demo early-return) so the
# pure-python `--demo` byte illustration runs anywhere, with no torch installed.
torch = None  # set in main()
dist = None  # set in main()

SENTINEL = 123.0  # not a plausible byte-swap of any expected value below
N = 1024


def classify(actual, expected):
    """Label a received 1-D tensor vs the expected fill value."""
    a = actual.float()
    if torch.allclose(a, torch.full_like(a, expected), atol=1e-3):
        return "PASS"
    if torch.allclose(a, torch.full_like(a, SENTINEL), atol=1e-3):
        return "FAIL/UNTOUCHED"
    return f"FAIL/WRONG(min={a.min().item():.4g}, max={a.max().item():.4g})"


def endianness_demo(rank: int) -> str:
    """Pure-python illustration of the spyre-comms rank-exchange byte bug.

    SocketOOB::setup_communication announced the rank with
    `write_on_socket(&myrank_, sizeof(int))` where myrank_ is uint64_t, while
    the listener reads a 4-byte `unsigned`. Shipping only the first sizeof(int)
    bytes of an 8-byte value sends the HIGH-order bytes on big-endian -> they
    are zero for any real rank -> the peer reads rank 0 ("Source rank and
    target rank are the same"). The fix sends the value as a fixed-width
    unsigned (the actual low bytes). No hardware needed; this is just bytes.
    """
    full8 = struct.pack("=Q", rank)  # native 8-byte uint64 == myrank_
    buggy_wire = full8[:4]  # write_on_socket(&myrank_, sizeof(int))
    buggy_read = struct.unpack("=I", buggy_wire)[0]  # listener reads 4-byte unsigned
    fixed_wire = struct.pack("=I", rank)  # static_cast<unsigned>(myrank_)
    fixed_read = struct.unpack("=I", fixed_wire)[0]
    return (
        f"  host byte order       : {sys.byteorder}-endian\n"
        f"  rank to announce      : {rank}\n"
        f"  myrank_ uint64 bytes  : {full8.hex(' ')}\n"
        f"  BUGGY wire (first 4B) : {buggy_wire.hex(' ')}  -> peer reads rank "
        f"{buggy_read}{'   <-- WRONG (big-endian truncation)' if buggy_read != rank else ''}\n"
        f"  FIXED wire (uint32)   : {fixed_wire.hex(' ')}  -> peer reads rank "
        f"{fixed_read}{'   <-- correct' if fixed_read == rank else ''}"
    )


def run_broadcast(device, world_size, rank):
    val = 7.0
    fill = val if rank == 0 else SENTINEL
    t = torch.full((N,), fill, dtype=torch.float16, device=device)
    dist.broadcast(t, src=0)
    return "PASS" if rank == 0 else classify(t.cpu(), val), f"src=0 sends {val}"


def run_send_recv(device, world_size, rank):
    if world_size != 2:
        return "SKIP", "assumes world_size==2"
    val = 5.0
    if rank == 0:
        dist.send(torch.full((N,), val, dtype=torch.float16, device=device), dst=1)
        return "PASS", "sender"
    t = torch.full((N,), SENTINEL, dtype=torch.float16, device=device)
    dist.recv(t, src=0)
    return classify(t.cpu(), val), f"rank0 -> rank1 sends {val}"


def run_all_reduce(device, world_size, rank):
    t = torch.full((N,), float(rank + 1), dtype=torch.float16, device=device)
    dist.all_reduce(t)  # SUM
    expected = float(sum(range(1, world_size + 1)))
    return classify(t.cpu(), expected), f"sum=={expected}"


def run_reduce(device, world_size, rank):
    t = torch.full((N,), float(rank + 1), dtype=torch.float16, device=device)
    dist.reduce(t, dst=0)  # SUM to rank 0
    if rank != 0:
        return "PASS", "non-root (value unspecified after reduce)"
    expected = float(sum(range(1, world_size + 1)))
    return classify(t.cpu(), expected), f"sum=={expected} on rank 0"


def run_all_gather(device, world_size, rank):
    t = torch.full((N,), float(rank + 1), dtype=torch.float16, device=device)
    out = [torch.full((N,), SENTINEL, dtype=torch.float16, device=device) for _ in range(world_size)]
    dist.all_gather(out, t)  # list form
    bad = [f"slot{r}:{s}" for r, o in enumerate(out) if (s := classify(o.cpu(), float(r + 1))) != "PASS"]
    return ("PASS" if not bad else "FAIL", "; ".join(bad) or "all slots correct")


def run_gather(device, world_size, rank):
    t = torch.full((N,), float(rank + 1), dtype=torch.float16, device=device)
    out = (
        [torch.full((N,), SENTINEL, dtype=torch.float16, device=device) for _ in range(world_size)]
        if rank == 0
        else None
    )
    dist.gather(t, out, dst=0)
    if rank != 0:
        return "PASS", "non-root sender"
    bad = [f"slot{r}:{s}" for r, o in enumerate(out) if (s := classify(o.cpu(), float(r + 1))) != "PASS"]
    return ("PASS" if not bad else "FAIL", "; ".join(bad) or "all slots correct")


# Primitives first (broadcast/send_recv) — the all_reduce fallback rides on them.
PROBES = [
    ("broadcast", run_broadcast),
    ("send_recv", run_send_recv),
    ("all_reduce", run_all_reduce),
    ("reduce", run_reduce),
    ("all_gather", run_all_gather),
    ("gather", run_gather),
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Print the endianness byte-level demo and exit (no hardware/torchrun needed).",
    )
    args, _ = parser.parse_known_args()
    if args.demo:
        for r in (0, 1):
            print(f"\n=== rank-exchange endianness demo for rank {r} ===\n{endianness_demo(r)}")
        return 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Show the byte-level reason BEFORE init, so it prints even on stock comms
    # where init_process_group dies at the rendezvous.
    print(
        f"\n[rank {rank}] rank-exchange wire bytes "
        f"(why stock rendezvous fails on big-endian):\n{endianness_demo(rank)}",
        flush=True,
    )

    # Lazy torch import (env above must be set first); bind as module globals so
    # the probe/classify functions resolve them.
    global torch, dist
    import torch
    import torch.distributed as dist

    import torch_spyre

    torch_spyre._autoload()
    torch.spyre.set_device(local_rank)

    # gloo for cpu tensors, spyreccl for spyre tensors. This triggers the OOB
    # rendezvous (createSpyreCCLBackend) — fails here on a stock comms .so.
    dist.init_process_group(
        backend="cpu:gloo,spyre:spyreccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    print(f"[rank {rank}] rendezvous OK — spyreccl group is up", flush=True)

    device = torch.device(f"spyre:{local_rank}")
    results = []
    for name, fn in PROBES:
        try:
            status, detail = fn(device, world_size, rank)
        except Exception as exc:  # noqa: BLE001 — capture every failure mode
            status = "UNIMPLEMENTED/ERROR"
            detail = f"{type(exc).__name__}: {exc}".splitlines()[0][:120]
        results.append((name, status, detail))
        try:
            dist.barrier()  # keep ranks loosely in lockstep between probes
        except Exception:  # noqa: BLE001
            pass

    width = max(len(n) for n, _, _ in results)
    report = [f"\n===== spyreccl collective report (rank {rank}/{world_size}) ====="]
    report += [f"  {n:<{width}}  {s:<22}  {d}" for n, s, d in results]
    report.append("=" * 60)
    print("\n".join(report), flush=True)

    dist.destroy_process_group()

    if rank == 0 and any(s.startswith("FAIL") for _, s, _ in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())