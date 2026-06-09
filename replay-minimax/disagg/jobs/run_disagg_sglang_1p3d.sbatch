#!/bin/bash
#SBATCH --job-name=mm25-disagg-sglang-1p3d
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Disaggregated 1P3D (SGLang) MiniMax-M2.5 serve + sweep on one GB300 node:
# 1 prefill GPU + 3 TP1 decode workers (GPU1/2/3), nixl transfer backend, official sglang
# router (one --decode per worker). EP n/a (TP1). See serve script header for the 1P3D
# metric caveats (sweep scrapes decode #1; recompute_steady can't form steady across 3).
#
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 sbatch run_disagg_sglang_1p3d.sbatch
set -x
MODEL="${MODEL:?set MODEL}"
IMAGE="${IMAGE:-/mnt/vast/squash_dupe/lmsysorg_sglang_nightly-dev-cu13-20260602-98a1b58c_arm64.sqsh}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"
DISAGG="/mnt/home/ppbhatt500/gpumode-triton/replay-minimax/disagg"

srun --container-image="$IMAGE" \
     --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
     --container-workdir="$DISAGG" \
     bash -lc "MODEL='$MODEL' CONCURRENCIES='$CONCURRENCIES' RESULT_DIR='${RESULT_DIR:-}' \
        bash '$DISAGG/serve_disagg_sglang_1p3d.sh'"
