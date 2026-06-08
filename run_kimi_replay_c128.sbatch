#!/bin/bash
#SBATCH --job-name=kimi-vllm-c128
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/home/ppbhatt500/gpumode-triton/results/kimi_b1/slurm-%j.out
#SBATCH --container-image=/mnt/vast/squash_dupe/vllm_vllm-openai_v0.21.0-ubuntu2404_arm64.sqsh
#SBATCH --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast
#SBATCH --container-workdir=/mnt/home/ppbhatt500/InferenceX

# High-concurrency colocated extension: c16,32,64,128 over the 150-session
# long-trace dataset (batch_long), warmup = first 15 sessions, per-conc
# Prometheus scrape -> c<conc>_prom.json.
set -x
nvidia-smi -L || true

cd /mnt/home/ppbhatt500/InferenceX
MODEL=/mnt/vast/models/kimi-k25-nvfp4 \
TP=4 \
DATASET=/mnt/home/ppbhatt500/gpumode-triton/replay/batch_long.replay.jsonl \
RESULT_DIR=/mnt/home/ppbhatt500/gpumode-triton/results/kimi_b1 \
CONCURRENCIES=4,8,16,32,64,128 \
WARMUP=1 \
WARMUP_SESSIONS=15 \
bash benchmarks/single_node/agentic/kimik2.5_fp4_b300_replay.sh
