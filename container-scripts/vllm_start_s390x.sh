#!/bin/bash -e

# AIU configuration is done at runtime by the base image's setup script.
# It runs AIU device discovery, generates the autogen senlib config, and
# prepends /opt/ibm/spyre/*/lib to LD_LIBRARY_PATH. The zz-*.sh scripts in
# /etc/profile.d/ run after this and have the last word on PATH/LD_LIBRARY_PATH
# ordering (our overlay) and (optionally) single-card override.
#
# `|| true`: this script runs under `set -e`, but ibm-aiu-setup.sh ends with a
# `chmod -R /tmp/etc` and a `cp ... /etc/aiu/senlib_config.json` that can return
# non-zero (e.g. pre-owned paths) without being fatal — the autogen config in
# /tmp/etc is what's actually used. Don't let a benign warning abort startup.
source /etc/profile.d/ibm-aiu-setup.sh || true

# Defense-in-depth: re-source our zz-prefixed overrides explicitly in case
# the parent shell didn't trigger profile.d.
[ -f /etc/profile.d/zz-flex-overlay.sh ]    && source /etc/profile.d/zz-flex-overlay.sh
[ -f /etc/profile.d/zz-aiu-single-card.sh ] && source /etc/profile.d/zz-aiu-single-card.sh

# VF (Virtual Function) execution is supported via the runtime's stream support
# (torch-spyre/flex main, ~2026-05-31 nightly toolchain — rebuilt into the base
# image). Select it with FLEX_DEVICE=VF (PF remains the default). The senlib
# config tweak below disables RISC-V emulation and is only required for VF —
# skip it under PF.
if [ "${FLEX_DEVICE:-PF}" = "VF" ]; then
  jq '(.SNT_MCI.DCR.MCI_CTRL.ENABLE_RISCV = "0x0") | del(.SNT_MCI.init) | (.METRICS.general.enable = true)' \
      "${AIU_AUTOGEN_SENLIB_CONFIG_FILE}" > "$HOME/.senlib.json"
fi

# Enable gcc-toolset-14 if present (provides matching libstdc++/libgcc on
# RHEL-stream bases that don't ship gcc-14 systemwide). On el10 the toolset
# is unnecessary and the enable file is absent — skip silently.
if [ -f /opt/rh/gcc-toolset-14/enable ]; then
  source /opt/rh/gcc-toolset-14/enable
fi

# Start the vLLM OpenAI API server. `exec` replaces the parent process so
# signals (SIGTERM from kubernetes / podman) are delivered to vllm directly.
exec python3 -m vllm.entrypoints.openai.api_server "$@"