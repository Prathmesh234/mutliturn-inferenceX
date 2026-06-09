#!/bin/bash
#SBATCH --job-name=mm25-agg-noep
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Aggregated (colocated) MiniMax-M2.5 on vLLM WITHOUT expert parallelism — the control
# for the EP comparison. Same one-node flow + identical 150-session sweep as run_agg.sbatch;
# only serve script differs (serve_agg_noep.sh, which drops --enable-expert-parallel).
# IMAGE pinned to the SAME v0.21.0 squash as the running EP job (6878) so the only
# variable is expert parallelism.
#
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 TP=4 \
#     CONCURRENCIES=4,8,16,32,64,128 sbatch run_agg_noep.sbatch
set -x
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${MODEL:?set MODEL}"; TP="${TP:-4}"
IMAGE="${IMAGE:-/mnt/vast/squash_dupe/vllm_vllm-openai_v0.21.0-ubuntu2404_arm64.sqsh}"
OFFLOAD_GB="${OFFLOAD_GB:-16}"; CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"
# RESULT_DIR is left for serve_agg_noep.sh to compute -> ../results_minimax/agg_vllm_tp<TP>_noep

srun --container-image="$IMAGE" \
     --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
     --container-workdir="$HERE" \
     bash -lc "MODEL='$MODEL' TP='$TP' OFFLOAD_GB='$OFFLOAD_GB' \
        CONCURRENCIES='$CONCURRENCIES' \
        DATASET='$HERE/batch_long.replay.jsonl' \
        bash '$HERE/serve_agg_noep.sh'"
