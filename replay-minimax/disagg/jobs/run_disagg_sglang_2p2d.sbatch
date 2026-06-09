#!/bin/bash
#SBATCH --job-name=mm25-disagg-sglang-2p2d
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Disaggregated 2P2D (SGLang) MiniMax-M2.5 serve + sweep on one GB300 node:
# TP2 prefill (GPU0,1) + TP2 decode (GPU2,3), nixl transfer backend, official sglang router.
# EP=1 (default) -> --ep-size 2 + flashinfer_cutlass; EP=0 is the pure-TP2 control.
#
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 EP=1 sbatch run_disagg_sglang_2p2d.sbatch
#   # no-EP control:  ... EP=0 sbatch run_disagg_sglang_2p2d.sbatch
set -x
MODEL="${MODEL:?set MODEL}"
EP="${EP:-1}"
IMAGE="${IMAGE:-/mnt/vast/squash_dupe/lmsysorg_sglang_nightly-dev-cu13-20260602-98a1b58c_arm64.sqsh}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"
DISAGG="/mnt/home/ppbhatt500/gpumode-triton/replay-minimax/disagg"

srun --container-image="$IMAGE" \
     --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
     --container-workdir="$DISAGG" \
     bash -lc "MODEL='$MODEL' EP='$EP' CONCURRENCIES='$CONCURRENCIES' RESULT_DIR='${RESULT_DIR:-}' \
        bash '$DISAGG/serve_disagg_sglang_2p2d.sh'"
