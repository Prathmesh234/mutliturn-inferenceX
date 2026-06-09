#!/bin/bash
#SBATCH --job-name=mm25-agg-bb
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
# DIAGNOSTIC: MiniMax-M2.5 aggregated vLLM TP+EP, big prefill/decode token budget,
# concurrency 128 only. Tests whether the ~800 tok/s/gpu ceiling is a serving cap
# (max-num-batched-tokens=8192) vs a model/roofline limit. Same dataset as 6878.
#
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 sbatch run_agg_bigbatch.sbatch
set -x
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${MODEL:-/mnt/vast/models/minimax-m2.5-nvfp4}"; TP="${TP:-4}"
IMAGE="${IMAGE:-/mnt/vast/squash_dupe/vllm_vllm-openai_v0.21.0-ubuntu2404_arm64.sqsh}"
CONCURRENCIES="${CONCURRENCIES:-128}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-32768}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

srun --container-image="$IMAGE" \
     --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
     --container-workdir="$HERE" \
     bash -lc "MODEL='$MODEL' TP='$TP' \
        CONCURRENCIES='$CONCURRENCIES' \
        MAX_NUM_BATCHED='$MAX_NUM_BATCHED' MAX_NUM_SEQS='$MAX_NUM_SEQS' \
        DATASET='$HERE/batch_long.replay.jsonl' \
        bash '$HERE/serve_agg_bigbatch.sh'"
