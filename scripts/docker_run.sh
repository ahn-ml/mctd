#!/usr/bin/env bash
# Run a command inside the C-MCTD Docker image with this repository mounted at /workspace.
# Usage: bash scripts/docker_run.sh python3 scripts/cmctd.py ...
# Environment: CMCTD_IMAGE (default cmctd), CMCTD_GPUS (GPU id(s) or "none", default 0),
#              CMCTD_DATASETS (OGBench dataset cache, default ~/.ogbench/data).
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${CMCTD_DATASETS:-$HOME/.ogbench/data}"
mkdir -p "$dataset_root"
dataset_root="$(cd -- "$dataset_root" && pwd)"
gpu_args=()
if [[ "${CMCTD_GPUS:-0}" != "none" ]]; then
  gpu_args=(--gpus "\"device=${CMCTD_GPUS:-0}\"")
fi
exec docker run --rm --init "${gpu_args[@]}" \
  --user "$(id -u):$(id -g)" --shm-size "${CMCTD_SHM_SIZE:-8g}" \
  --tmpfs /tmp/cmctd-home:rw,mode=1777 \
  -e HOME=/tmp/cmctd-home -e MUJOCO_GL=egl -e WANDB_MODE=disabled \
  --mount "type=bind,src=$repo_root,dst=/workspace" \
  --mount "type=bind,src=$dataset_root,dst=/tmp/cmctd-home/.ogbench/data" \
  -w /workspace "${CMCTD_IMAGE:-cmctd}" "$@"
