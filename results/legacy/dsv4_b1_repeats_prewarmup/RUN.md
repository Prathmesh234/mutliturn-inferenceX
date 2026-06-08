# DeepSeek-V4 FP4 — replay pareto run

Model: `/mnt/vast/tom.chen/dsv4` (fp4, 64 shards, TP=4) · vLLM 0.21.0 container · single GB300 node.

## Submit (sbatch — what we actually use)
```bash
sbatch /mnt/home/ppbhatt500/gpumode-triton/run_dsv4_replay.sbatch
```
The sbatch wrapper pins the container + mounts and calls the launcher with:
`MODEL=/mnt/vast/tom.chen/dsv4 TP=4 CONCURRENCIES=1,4,16 DURATION=60 WARMUP=15`.

## Equivalent interactive srun (one line)
```bash
srun --partition=gb300 --nodes=1 --exclusive --gres=gpu:4 --mem=800G --time=03:00:00 \
  --container-image=/mnt/vast/squash_dupe/vllm_vllm-openai_v0.21.0-ubuntu2404_arm64.sqsh \
  --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
  --container-workdir=/mnt/home/ppbhatt500/InferenceX \
  --pty bash -lc '
    MODEL=/mnt/vast/tom.chen/dsv4 TP=4 \
    DATASET=/mnt/home/ppbhatt500/gpumode-triton/replay/batch_1.replay.jsonl \
    RESULT_DIR=/mnt/home/ppbhatt500/gpumode-triton/results/dsv4_b1 \
    CONCURRENCIES=1,4,16 DURATION=60 WARMUP=15 \
    bash benchmarks/single_node/agentic/dsv4_fp4_vllm_replay.sh'
```

## Outputs (this folder)
`pareto.csv`, `pareto.png`, `conc<N>.json` (one per concurrency), `server.log`, `gpu_metrics.csv`.

To sweep the full range, set e.g. `CONCURRENCIES=1,2,4,8,16,32,64,128` (server `--max-num-seqs` auto-tracks the max).
