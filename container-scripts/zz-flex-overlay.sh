# shellcheck shell=sh
# Sources after /etc/profile.d/ibm-aiu-setup.sh (zz- prefix sorts last) and
# re-prepends the rebuilt flex/deeptools/libaiupti/spyre-comms libs from the
# torch-spyre-docs build pipeline so they win over the base image's
# /opt/ibm/spyre/*/lib RPM versions at dynamic-link time. Without this,
# libflex.so resolves to the older d19074e5 RPM build which lacks the
# 3-arg RuntimeOperationCompute / setPipelineBarrier symbols torch_spyre
# was compiled against.
: "${SENTIENT_BASE_INSTALL_DIR:=/workspace/dt-inductor/sentient}"
export LD_LIBRARY_PATH="${SENTIENT_BASE_INSTALL_DIR}/runtime/lib:${SENTIENT_BASE_INSTALL_DIR}/deeptools/lib:${SENTIENT_BASE_INSTALL_DIR}/libaiupti/lib:${SENTIENT_BASE_INSTALL_DIR}/spyre_comms/lib:${LD_LIBRARY_PATH}"