#!/bin/bash
# Launch the 7 disagg jobs (150-session batch_long, conc 4..128, warmup 15,
# per-run Prometheus scrape). Each driver is setsid-detached so it survives the
# login shell; the Slurm jobs survive regardless. Staggered to avoid srtctl/etcd
# submit contention (per-job UV_CACHE_DIR already isolates the uv cache).
set -u
RES=/mnt/home/ppbhatt500/gpumode-triton/results
DR=$HOME/disagg_runs

launch() {
  local name="$1" engine="$2" recipe="$3"
  local ws="$DR/$name"
  cd "$ws" || { echo "MISSING WORKSPACE $ws"; return 1; }
  local fw_env=""
  if [ "$engine" = "vllm" ]; then
    fw_env="FRAMEWORK=dynamo-vllm IS_AGENTIC=1"
  else
    fw_env="FRAMEWORK=dynamo-sglang"
  fi
  echo "[launch] $name ($engine)  recipe=$recipe"
  setsid env UV_CACHE_DIR=/tmp/uvcache-$name MODEL_PREFIX=dsv4 PRECISION=fp4 \
      $fw_env \
      CONFIG_FILE="$recipe" \
      GITHUB_WORKSPACE="$PWD" RUNNER_NAME="$name" \
      SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
      ISL=8192 OSL=1024 \
      bash runners/launch_gb300-cw.sh > "$RES/$name.driver.log" 2>&1 < /dev/null &
  echo "  driver pid $! -> $RES/$name.driver.log"
}

V=recipes/vllm/deepseek-v4/agentic
S=recipes/sglang/deepseek-v4/agentic

launch dsv4_vllm_4p_4d    vllm   "$V/disagg-gb300-1p1d-agentic.yaml";          sleep 30
launch dsv4_vllm_8p_4d    vllm   "$V/disagg-gb300-2p1d-agentic.yaml";          sleep 30
launch dsv4_vllm_4p_8d    vllm   "$V/disagg-gb300-1p2d-agentic.yaml";          sleep 30
launch dsv4_vllm_4p_12d   vllm   "$V/disagg-gb300-1p3d-agentic.yaml";          sleep 30
launch dsv4_sglang_4p_4d  sglang "$S/disagg-gb300-1p1d-tp4-tp4-agentic.yaml";  sleep 30
launch dsv4_sglang_4p_8d  sglang "$S/disagg-gb300-1p2d-dep4-dep8-agentic.yaml"; sleep 30
launch dsv4_sglang_4p_12d sglang "$S/disagg-gb300-1p3d-dep4-dep12-agentic.yaml"
echo "[launch] all 7 drivers started."
