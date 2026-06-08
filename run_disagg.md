# Disaggregated-serving launch commands (DSV4 FP4, GB300, 4 GPU/node)

6 recipes = 3 P/D topologies (A/B/C) x 2 engines (sglang, vllm), each driven
by our custom replay client (`benchmarks/multi_node/agentic_replay_custom.sh`).

**DO NOT run these yet** — commands only, for review.

## How the driver maps FRAMEWORK -> recipe dir + CONFIG_FILE

From `runners/launch_gb300-cw.sh`:

- The launcher `git clone`s srt-slurm into `./srt-slurm`, `cd`s into it, then
  `cp -rT "$SRT_RECIPE_SRC" "$SRT_RECIPE_DST"` overlays our hand-rolled recipes
  onto the checkout, and finally runs `srtctl apply -f "$CONFIG_FILE"` **from
  inside `./srt-slurm`**. So `CONFIG_FILE` is a path **relative to the
  `srt-slurm` clone root**, pointing at the overlaid copy under `recipes/...`.
- Branch selection (lines 20-49):
  - `IS_AGENTIC=1` (dsv4/fp4):  SRC = `.../vllm/deepseek-v4/agentic`,
    DST = `recipes/vllm/deepseek-v4/agentic`  (pins upstream SHA
    `127597c2926467db06e6707e0aa9227261c6c02a`). **Use this for the vLLM recipes.**
  - `FRAMEWORK=dynamo-sglang` (no IS_AGENTIC): SRC = `.../sglang/deepseek-v4`,
    DST = `recipes/sglang/deepseek-v4` (ref `main`). The overlay copies the
    whole `sglang/deepseek-v4` tree, **including our new `agentic/` subdir**, so
    our sglang agentic recipes land at `recipes/sglang/deepseek-v4/agentic/...`.
    **Use this (NOT IS_AGENTIC) for the sglang recipes.**
- `srtslurm.yaml` is generated with `gpus_per_node: 4`; account `cw-sup`,
  partition `all` are forced by the script regardless of what you pass, but
  passing them is harmless.

## Node counts per job

| recipe        | engine | prefill          | decode           | infra | total nodes |
|---------------|--------|------------------|------------------|-------|-------------|
| A 1p1d        | sglang | 1 node / 4 GPU   | 1 node / 4 GPU   |  0    | 2           |
| B 2p1d        | sglang | 2 nodes / 4 GPU ea (8) | 1 node / 4 GPU |  0 | 3           |
| C 1p2d        | sglang | 1 node / 4 GPU   | 2 nodes / 8 GPU  |  0    | 3           |
| A 1p1d        | vllm   | 1 node / 4 GPU   | 1 node / 4 GPU   |  1    | 3           |
| B 2p1d        | vllm   | 2 nodes / 4 GPU ea (8) | 1 node / 4 GPU |  1 | 4           |
| C 1p2d        | vllm   | 1 node / 4 GPU   | 2 nodes / 8 GPU  |  1    | 4           |

(vLLM topologies add +1 dedicated NATS/etcd infra node via
`infra.etcd_nats_dedicated_node: true`.)

## SGLang recipes (FRAMEWORK=dynamo-sglang, NO IS_AGENTIC)

```bash
# A — 1 prefill TP4 + 1 decode TP4 (2 nodes)
MODEL_PREFIX=dsv4 PRECISION=fp4 FRAMEWORK=dynamo-sglang \
  CONFIG_FILE=recipes/sglang/deepseek-v4/agentic/disagg-gb300-1p1d-tp4-tp4-agentic.yaml \
  GITHUB_WORKSPACE=$PWD RUNNER_NAME=manual SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
  bash runners/launch_gb300-cw.sh

# B — 2 prefill TP4 + 1 decode TP4 (3 nodes)
MODEL_PREFIX=dsv4 PRECISION=fp4 FRAMEWORK=dynamo-sglang \
  CONFIG_FILE=recipes/sglang/deepseek-v4/agentic/disagg-gb300-2p1d-tp4-tp4-agentic.yaml \
  GITHUB_WORKSPACE=$PWD RUNNER_NAME=manual SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
  bash runners/launch_gb300-cw.sh

# C — 1 prefill DEP4 + 1 decode DEP8 (2-node decode) (3 nodes)
MODEL_PREFIX=dsv4 PRECISION=fp4 FRAMEWORK=dynamo-sglang \
  CONFIG_FILE=recipes/sglang/deepseek-v4/agentic/disagg-gb300-1p2d-dep4-dep8-agentic.yaml \
  GITHUB_WORKSPACE=$PWD RUNNER_NAME=manual SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
  bash runners/launch_gb300-cw.sh
```

## vLLM recipes (FRAMEWORK=dynamo-vllm, IS_AGENTIC=1)

`IS_AGENTIC=1` is what selects the `vllm/deepseek-v4/agentic` source dir (lines
20-35). FRAMEWORK value is otherwise unused on the agentic branch, but set it
to `dynamo-vllm` for clarity.

```bash
# A — 1 prefill DEP4 + 1 decode DEP4 (+1 infra = 3 nodes)
MODEL_PREFIX=dsv4 PRECISION=fp4 FRAMEWORK=dynamo-vllm IS_AGENTIC=1 \
  CONFIG_FILE=recipes/vllm/deepseek-v4/agentic/disagg-gb300-1p1d-agentic.yaml \
  GITHUB_WORKSPACE=$PWD RUNNER_NAME=manual SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
  bash runners/launch_gb300-cw.sh

# B — 2 prefill DEP4 + 1 decode DEP4 (+1 infra = 4 nodes)
MODEL_PREFIX=dsv4 PRECISION=fp4 FRAMEWORK=dynamo-vllm IS_AGENTIC=1 \
  CONFIG_FILE=recipes/vllm/deepseek-v4/agentic/disagg-gb300-2p1d-agentic.yaml \
  GITHUB_WORKSPACE=$PWD RUNNER_NAME=manual SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
  bash runners/launch_gb300-cw.sh

# C — 1 prefill DEP4 + 1 decode DEP8 (2-node decode) (+1 infra = 4 nodes)
MODEL_PREFIX=dsv4 PRECISION=fp4 FRAMEWORK=dynamo-vllm IS_AGENTIC=1 \
  CONFIG_FILE=recipes/vllm/deepseek-v4/agentic/disagg-gb300-1p2d-agentic.yaml \
  GITHUB_WORKSPACE=$PWD RUNNER_NAME=manual SLURM_ACCOUNT=cw-sup SLURM_PARTITION=all \
  bash runners/launch_gb300-cw.sh
```

## Notes / VERIFY

- `RUNNER_NAME=manual` overwrites each recipe's `name:` field at submit time
  (launcher lines 271-277). The `name:` we set in each file is for readability;
  the submitted job name becomes `manual`.
- The custom hook runs *inside the container* as
  `bash /infmax-workspace/benchmarks/multi_node/agentic_replay_custom.sh`
  (workspace mounted at `/infmax-workspace`). The driver's `default_mounts` only
  mount `/configs/dynamo-wheels` + `/aiperf_mmap_cache` — **`/mnt/home` is NOT
  mounted** in the benchmark container. So the shim now reads the replayer +
  dataset from the workspace mount (`/infmax-workspace/utils/custom_replay/`,
  incl. the staged `batch_3x.replay.jsonl`) and writes to the managed `/logs`.
- The shim **auto-discovers** the live endpoint (probes `/v1/models` on the
  PORT env then 8000/8888/8080/9000/8001) and uses the model id the frontend
  advertises — so it works whether the dynamo/nginx frontend binds 8000
  (vllm) or another port (sglang), with no hardcoding.

## Collect results after each job

Results land in the container's `/logs/agentic`, which srtctl collects to
`srt-slurm/outputs/<JOB_ID>/logs/agentic/` on the host. After a job finishes:

```bash
JOB=<job_id>; COMBO=<engine_combo>     # e.g. COMBO=sglang_1p1d
cp -r ~/InferenceX/srt-slurm/outputs/$JOB/logs/agentic \
      ~/gpumode-triton/results/dsv4_disagg_$COMBO
cat ~/gpumode-triton/results/dsv4_disagg_$COMBO/pareto.csv
```
(Exact `outputs/` path may vary by srtctl version — if absent, `find ~/InferenceX/srt-slurm -name pareto.csv` after the run.)
