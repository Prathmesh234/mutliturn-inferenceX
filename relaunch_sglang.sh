#!/bin/bash
set -u
RES=/mnt/home/ppbhatt500/gpumode-triton/results
DR=$HOME/disagg_runs
IMG="lmsysorg/sglang:nightly-dev-cu13-20260602-98a1b58c"
S=recipes/sglang/deepseek-v4/agentic
launch() {
  local name="$1" recipe="$2"
  cd "$DR/$name" || { echo "MISSING $name"; return 1; }
  echo "[relaunch] $name  recipe=$recipe"
  setsid env UV_CACHE_DIR=/tmp/uvcache-$name MODEL_PREFIX=dsv4 PRECISION=fp4 \
      FRAMEWORK=dynamo-sglang IMAGE="$IMG" CONFIG_FILE="$recipe" \
      GITHUB_WORKSPACE="$PWD" RUNNER_NAME="$name" SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
      ISL=8192 OSL=1024 \
      bash runners/launch_gb300-cw.sh > "$RES/$name.driver.log" 2>&1 < /dev/null &
  echo "  driver pid $!"
}
launch dsv4_sglang_4p_4d  "$S/disagg-gb300-1p1d-tp4-tp4-agentic.yaml";   sleep 30
launch dsv4_sglang_4p_8d  "$S/disagg-gb300-1p2d-dep4-dep8-agentic.yaml"; sleep 30
launch dsv4_sglang_4p_12d "$S/disagg-gb300-1p3d-dep4-dep12-agentic.yaml"
echo "[relaunch] 3 sglang drivers started (radix cache ENABLED)."
