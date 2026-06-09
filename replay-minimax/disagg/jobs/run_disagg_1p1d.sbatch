#!/bin/bash
#SBATCH --job-name=mm25-disagg-1p1d
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Disaggregated 1P1D (vLLM) MiniMax-M2.5 serve + sweep on one GB300 node:
# 1 prefill GPU + 1 decode GPU, P2pNcclConnector, official in-image proxy. EP n/a (TP1).
#
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 sbatch run_disagg_1p1d.sbatch
#   # toy: MODEL=... CONCURRENCIES=32 RESULT_DIR=<...>/disagg_vllm_1p1d_toy \
#   #        sbatch --time=00:45:00 run_disagg_1p1d.sbatch
set -x
MODEL="${MODEL:?set MODEL}"
IMAGE="${IMAGE:-/mnt/vast/squash_dupe/vllm_vllm-openai_v0.21.0-ubuntu2404_arm64.sqsh}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"
DISAGG="/mnt/home/ppbhatt500/gpumode-triton/replay-minimax/disagg"

srun --container-image="$IMAGE" \
     --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
     --container-workdir="$DISAGG" \
     bash -lc "MODEL='$MODEL' CONCURRENCIES='$CONCURRENCIES' RESULT_DIR='${RESULT_DIR:-}' \
        bash '$DISAGG/serve_disagg_1p1d.sh'"
