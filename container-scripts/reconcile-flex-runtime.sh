#!/usr/bin/env bash
#
# reconcile-flex-runtime.sh — guarantee a SINGLE, ABI-correct libflex.so in the
# image so torch_spyre/_C.so always binds the flex it was compiled against,
# regardless of LD_LIBRARY_PATH ordering.
#
# Why this exists
# ---------------
# The base image (icr.io/ocp-ai-on-z/spyre-granite-eval) ships a STALE flex
# under /opt/ibm/spyre/runtime/lib. It exports the NON-const overload
#     flex::RuntimeOperationD2H::setPipelineBarrier(bool)          (mangled _ZN…)
# The torch-spyre-docs build pipeline (Dockerfile.base.s390x) rebuilds the
# matching flex into the overlay at ${SPYRE_OVERLAY_DIR}/runtime/lib, which
# exports the CONST overload
#     flex::RuntimeOperationD2H::setPipelineBarrier(bool) const    (mangled _ZNK…)
# torch_spyre/_C.so references the const symbol (U _ZNK4flex19RuntimeOperationD2H18setPipelineBarrierEb).
#
# The base image's /etc/profile.d/ibm-aiu-setup.sh prepends /opt/ibm/spyre/*/lib
# to LD_LIBRARY_PATH on login shells. When that wins, the loader binds the stale
# non-const libflex and the import dies with:
#     ImportError: …/torch_spyre/_C.so: undefined symbol:
#       _ZNK4flex19RuntimeOperationD2H18setPipelineBarrierEb
#     RuntimeError: Failed to load the backend extension: torch_spyre
#
# Fix: physically delete every libflex.so* that is NOT the overlay build. With
# only one libflex on the system, load order can never reintroduce the stale
# ABI. ONLY libflex* is touched — senlib, libflightlog, deeptools, spyre_comms,
# and the AIU discovery scripts under /opt/ibm/spyre are left intact (we do NOT
# `rm -rf` the directory).
#
# Idempotent and safe to run in both the toolchain (base) and app images, and a
# second time as a pure guard. Exits non-zero if the single-const-flex invariant
# cannot be met, which fails the Docker build.
set -euo pipefail

OVERLAY="${SPYRE_OVERLAY_DIR:-${SENTIENT_BASE_INSTALL_DIR:-/workspace/dt-inductor/sentient}}"
CANON="${OVERLAY}/runtime/lib/libflex.so"
CONST_SYM='setPipelineBarrier(bool) const'

log()  { printf '[reconcile-flex] %s\n' "$*"; }
die()  { printf '[reconcile-flex] FATAL: %s\n' "$*" >&2; exit 1; }

# Does $1 (.so) export the const setPipelineBarrier overload?
# NOTE: do NOT inline this as `nm ... | grep -q ...`. Under `set -o pipefail`,
# `grep -q` exits on first match and closes the pipe; `nm` then dies with
# SIGPIPE (141), and pipefail reports the pipeline as failed even though the
# symbol IS present — a false negative. Capture nm's output first, then grep a
# here-string (no pipe, no SIGPIPE).
flex_exports_const() {
  local out
  out="$(nm -DC "$1" 2>/dev/null || true)"
  grep -qF "${CONST_SYM}" <<<"${out}"
}

command -v nm   >/dev/null 2>&1 || die "nm not found (binutils required)"
command -v find >/dev/null 2>&1 || die "find not found"

find_flex() {
  find / \( -path /proc -o -path /sys -o -path /dev \) -prune -o \
      -name 'libflex.so*' -print 2>/dev/null
}

# 1. The overlay flex must exist and export the CONST setPipelineBarrier symbol
#    that torch_spyre/_C.so was compiled against.
[ -e "${CANON}" ] || die "overlay flex not found at ${CANON}"
if ! flex_exports_const "${CANON}"; then
  log "setPipelineBarrier symbols present in ${CANON}:"
  nm -DC "${CANON}" 2>/dev/null | grep -i setpipelinebarrier >&2 || true
  die "overlay flex ${CANON} does not export the const '${CONST_SYM}'"
fi
canon_real="$(readlink -f "${CANON}")"
log "canonical (const-ABI) flex: ${CANON} -> ${canon_real}"

# 2. Remove every OTHER libflex.so* on the system. Keep anything inside the
#    overlay tree and anything that already resolves to the canonical file.
removed=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in "${OVERLAY}/"*) continue ;; esac
  [ "$(readlink -f "$f")" = "${canon_real}" ] && continue
  sym="$(nm -DC "$f" 2>/dev/null | grep -i setpipelinebarrier | head -1 | sed 's/^[[:space:]]*//')"
  log "removing stale flex: $f  [${sym:-symbol unknown}]"
  rm -f "$f"
  removed=$((removed + 1))
done < <(find_flex)
log "removed ${removed} stale libflex file(s)"

# 3. Post-condition guard: exactly ONE distinct libflex.so* remains and it is
#    the const build. Fail the build otherwise.
mapfile -t survivors < <(find_flex | xargs -r -n1 readlink -f | sort -u)
if [ "${#survivors[@]}" -ne 1 ]; then
  log "remaining distinct libflex copies (${#survivors[@]}):"
  printf '  %s\n' "${survivors[@]}" >&2
  die "expected exactly 1 libflex.so after reconcile, found ${#survivors[@]}"
fi
flex_exports_const "${survivors[0]}" \
  || die "surviving flex ${survivors[0]} lacks the const '${CONST_SYM}'"

# 4. Best-effort: if torch_spyre is importable in this stage, confirm _C.so has
#    no unresolved libflex dependency. Skipped silently if the package or ldd is
#    unavailable (e.g. a stage before the venv is in place).
cso="$(python3 - <<'PY' 2>/dev/null || true
import importlib.util, os
spec = importlib.util.find_spec("torch_spyre")
if spec and spec.submodule_search_locations:
    d = list(spec.submodule_search_locations)[0]
    for n in sorted(os.listdir(d)):
        if n.startswith("_C") and n.endswith(".so"):
            print(os.path.join(d, n)); break
PY
)"
if [ -n "${cso}" ] && command -v ldd >/dev/null 2>&1; then
  if ldd "${cso}" 2>/dev/null | grep -i 'libflex' | grep -qi 'not found'; then
    log "$(ldd "${cso}" 2>/dev/null | grep -i flex)"
    die "torch_spyre _C.so cannot resolve libflex against the shipped runtime"
  fi
  log "_C.so flex linkage: $(ldd "${cso}" 2>/dev/null | grep -i flex | sed 's/^[[:space:]]*//')"
fi

log "OK — single const-ABI flex: ${survivors[0]}"