# shellcheck shell=sh
# Sources after /etc/profile.d/ibm-aiu-setup.sh (zz- prefix sorts last). That
# base script repoints PATH/LD_LIBRARY_PATH at the base image's /opt/ibm/spyre
# RPM tree (deriving from SENTIENT_BASE_INSTALL_DIR=/opt/ibm/spyre, which it
# needs to find aiu-discover-topo / aiu-assign-ranks.py / senlib). This script
# re-prepends our rebuilt flex/deeptools/libaiupti/spyre-comms from the
# torch-spyre-docs build pipeline so they win at dynamic-link AND exec time:
#
#   - LD_LIBRARY_PATH: without this, libflex.so resolves to the older base RPM
#     build (d19074e5) which lacks the symbols torch_spyre was compiled against.
#   - PATH: without this, dxp_standalone (the kernel bundler invoked during AIU
#     graph compilation) resolves to the base RPM binary while our overlay libs
#     are on LD_LIBRARY_PATH — that binary/lib mismatch aborts with
#     std::out_of_range (map::at).
#
# Keyed off SPYRE_OVERLAY_DIR (set in the release image) — NOT
# SENTIENT_BASE_INSTALL_DIR, which now points at the base RPM tree.
: "${SPYRE_OVERLAY_DIR:=/workspace/dt-inductor/sentient}"
export LD_LIBRARY_PATH="${SPYRE_OVERLAY_DIR}/runtime/lib:${SPYRE_OVERLAY_DIR}/deeptools/lib:${SPYRE_OVERLAY_DIR}/libaiupti/lib:${SPYRE_OVERLAY_DIR}/spyre_comms/lib:${LD_LIBRARY_PATH}"
export PATH="${SPYRE_OVERLAY_DIR}/runtime/bin:${SPYRE_OVERLAY_DIR}/deeptools/bin:${SPYRE_OVERLAY_DIR}/libaiupti/bin:${SPYRE_OVERLAY_DIR}/spyre_comms/bin:${PATH}"