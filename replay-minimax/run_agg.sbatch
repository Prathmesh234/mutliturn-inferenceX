#!/bin/bash
#SBATCH --job-name=mm25-agg
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Aggregated (colocated) MiniMax-M2.5 serve + concurrency sweep on one GB300 node.
# Stage 1-2 by default; stage 3 = pass OFFLOAD_GB=<N>.
#
# Submit (set MODEL + IMAGE to real values first):
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 TP=4 IMAGE=<vllm-arm64-squash-or-docker> \
#     CONCURRENCIES=4,8,16,32,64,128 sbatch run_agg.sbatch
#   # stage 3:  ... OFFLOAD_GB=40 sbatch run_agg.sbatch
set -x
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${MODEL:?set MODEL}"; TP="${TP:-4}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.1-aarch64-ubuntu2404}"
OFFLOAD_GB="${OFFLOAD_GB:-0}"; CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"
# RESULT_DIR is left for serve_agg.sh to compute -> ../results_minimax/agg_vllm_tp<TP>[_offloadN]

srun --container-image="$IMAGE" \
     --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
     --container-workdir="$HERE" \
     bash -lc "MODEL='$MODEL' TP='$TP' OFFLOAD_GB='$OFFLOAD_GB' \
        CONCURRENCIES='$CONCURRENCIES' \
        DATASET='$HERE/batch_long.replay.jsonl' \
        bash '$HERE/serve_agg.sh'"
