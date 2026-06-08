#!/bin/bash
#SBATCH --job-name=dsv4-replay
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/ppbhatt500/gpumode-triton/results/dsv4_b1/slurm-%j.out
#SBATCH --container-image=/mnt/vast/squash_dupe/vllm_vllm-openai_v0.21.0-ubuntu2404_arm64.sqsh
#SBATCH --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast
#SBATCH --container-workdir=/mnt/home/ppbhatt500/InferenceX

set -x
nvidia-smi -L || true

cd /mnt/home/ppbhatt500/InferenceX
MODEL=/mnt/vast/tom.chen/dsv4 \
TP=4 \
DATASET=/mnt/home/ppbhatt500/gpumode-triton/replay/batch_3x.replay.jsonl \
RESULT_DIR=/mnt/home/ppbhatt500/gpumode-triton/results/dsv4_b1 \
CONCURRENCIES=4,8,16 \
WARMUP=1 \
bash benchmarks/single_node/agentic/dsv4_fp4_vllm_replay.sh
