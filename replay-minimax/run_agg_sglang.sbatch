#!/bin/bash
#SBATCH --job-name=mm25-agg-sglang
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Aggregated MiniMax-M2.5 on SGLang — same one-node colocated flow as run_agg.sbatch,
# on a DIFFERENT node, identical 150-session sweep + pre-arrange + prom + GPU metrics.
# Only the serve script differs (serve_agg_sglang.sh).
#
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 TP=4 IMAGE=<sglang-arm64-squash-or-docker> \
#     CONCURRENCIES=4,8,16,32,64,128 sbatch run_agg_sglang.sbatch
set -x
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${MODEL:?set MODEL}"; TP="${TP:-4}"
IMAGE="${IMAGE:-/mnt/vast/squash_dupe/lmsysorg_sglang_nightly-dev-cu13-20260602-98a1b58c_arm64.sqsh}"
OFFLOAD_GB="${OFFLOAD_GB:-0}"; CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"

srun --container-image="$IMAGE" \
     --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
     --container-workdir="$HERE" \
     bash -lc "MODEL='$MODEL' TP='$TP' OFFLOAD_GB='$OFFLOAD_GB' \
        CONCURRENCIES='$CONCURRENCIES' \
        DATASET='$HERE/batch_long.replay.jsonl' \
        bash '$HERE/serve_agg_sglang.sh'"
