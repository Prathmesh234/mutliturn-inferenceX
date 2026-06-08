# Checkpoint — agentic trace replay pareto (read this to resume)

_Last updated: 2026-06-08, mid-run. Also see memory files: `inferencex-agentic-replay`,
`gpumode-triton-project`, `gb300-cluster`, `no-dummy-results`._

---
# ========== HANDOFF (2026-06-08 ~01:30 UTC) — READ FIRST ==========

## TL;DR for the next agent
Two tracks in flight: (A) **colocated** replay pareto (4 sbatch jobs, the original
deliverable) and (B) **disaggregated** (prefill/decode-split) serving via NVIDIA
srt-slurm + Dynamo (6 jobs, NEW, hard-won). User left (wifi off). All jobs are
real Slurm jobs that survive the session and write results to disk; the disagg
**babysitter drivers run in the (now-dead) login shell** so they may have died —
**that does NOT kill the Slurm jobs**, which keep running and writing to
`results/<name>/`. First action on resume: `squeue -u ppbhatt500` and read results.

## How to check everything
```bash
squeue -u ppbhatt500 -o "%.8i %.24j %.8T %.6D %.10M"
# colocated results:
for d in dsv4_b1 kimi_b1 dsv4_b1_sglang kimi_b1_sglang; do echo "== $d =="; cat ~/gpumode-triton/results/$d/pareto.csv 2>/dev/null; done
# disagg results (write DIRECTLY here via the /mnt/home mount):
for d in dsv4_sglang_4p_4d dsv4_sglang_8p_4d dsv4_sglang_4p_8d dsv4_vllm_4p_4d dsv4_vllm_8p_4d dsv4_vllm_4p_8d; do
  echo "== $d =="; ls ~/gpumode-triton/results/$d/ 2>/dev/null; cat ~/gpumode-triton/results/$d/pareto.csv 2>/dev/null; done
```

## (A) Colocated jobs (sbatch, the proven path) — job IDs at handoff
- 6717 vllm-dsv4 -> results/dsv4_b1 ; 6718 vllm-kimi -> results/kimi_b1
- 6720 sglang-kimi -> results/kimi_b1_sglang ; 6723 sglang-dsv4 -> results/dsv4_b1_sglang
- At handoff: kimi jobs on conc-8 (conc4.json+conc8.json written); dsv4 jobs on
  conc-4 (slow — see "why dsv4 slow"). Each will write conc4/8/16.json + pareto.csv.
- These use the 60-session dataset `replay/batch_3x.replay.jsonl` + `--warmup` + conc 4,8,16.
- Relaunch (if needed): `sbatch run_{dsv4,kimi}_replay.sbatch` / `run_{dsv4,kimi}_sglang_replay.sbatch`.

## (B) Disagg serving (NEW) — what it is + job IDs at handoff
6 jobs = 3 P/D topologies x {sglang, vllm}, DSv4 FP4, GB300. Driven by NVIDIA
srt-slurm `srtctl` + Dynamo frontend; our 60-session replay sweep is the client.
Submitted at handoff (RUNNING, survive session): **6777** dsv4_sglang_4p_4d (2 nodes),
**6780** dsv4_sglang_8p_4d (3), **6783** dsv4_sglang_4p_8d (3), **6786** dsv4_vllm_4p_4d (3).
Still in driver-setup at handoff (may NOT have submitted before wifi died — CHECK squeue
for names dsv4_vllm_8p_4d / dsv4_vllm_4p_8d; if absent, relaunch them):
dsv4_vllm_8p_4d, dsv4_vllm_4p_8d.

Topologies (GPU counts; GB300=4gpu/node): 4p_4d=1P1D(2 nodes), 8p_4d=2P1D(3),
4p_8d=1P2D(3). vllm jobs also use +1 etcd/NATS infra node.

### How disagg is launched (NOT a plain sbatch)
Per job, an isolated workspace copy lives at `~/disagg_runs/<name>/` (rsync of
`~/InferenceX`, 5.4MB each). The driver `runners/launch_gb300-cw.sh` clones
srt-slurm into that workspace, installs srtctl, `srtctl apply`s the recipe (submits
ONE Slurm job), then BABYSITS until done. Launch command per job (from its workspace):
```bash
cd ~/disagg_runs/<name>
UV_CACHE_DIR=/tmp/uvcache-<name> MODEL_PREFIX=dsv4 PRECISION=fp4 \
  FRAMEWORK=dynamo-sglang \            # vllm: FRAMEWORK=dynamo-vllm IS_AGENTIC=1
  IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260602-98a1b58c \   # vllm: vllm/vllm-openai:v0.21.0-ubuntu2404
  CONFIG_FILE=recipes/<engine>/deepseek-v4/agentic/<recipe>.yaml \
  GITHUB_WORKSPACE=$PWD RUNNER_NAME=<name> ISL=8192 OSL=1024 \
  bash runners/launch_gb300-cw.sh > ~/gpumode-triton/results/<name>.driver.log 2>&1 &
```
Recipes (in InferenceX, source of truth): `benchmarks/multi_node/srt-slurm-recipes/
{sglang,vllm}/deepseek-v4/agentic/disagg-gb300-*-agentic.yaml`. Client shim:
`benchmarks/multi_node/agentic_replay_custom.sh` (runs `utils/custom_replay/sweep_pareto.py`
against the Dynamo frontend `localhost:8000`, auto-discovers port+model-id, writes
to RESULT_DIR=/mnt/home/.../results/<name>, also scrapes `/metrics`->server_metrics.prom).
Launch doc: `~/gpumode-triton/run_disagg.md`.

### The disagg FIX CHAIN (each failure found by running; all now applied to source + all 6 workspaces)
1. **flashinfer_mxfp4**: sglang DSv4 FP4 MoE crashed "Hidden size mismatch" in the
   triton fp8 MoE -> added `--moe-runner-backend flashinfer_mxfp4` (in recipes).
2. **container 20260520 -> 20260602**: old sglang nightly gave decode
   "bootstrap_info is required" + flash-mla crash; newer `nightly-dev-cu13-20260602-98a1b58c`
   fixed it. All 3 sglang recipes pin it. (vllm uses vllm-openai v0.21.0.)
3. **vllm x86 .venv leak**: login node is x86, compute is aarch64; the vllm-ref
   job's uv grabbed the login-built x86 `.venv` -> "Exec format error". Fixes in
   `launch_gb300-cw.sh`: `unset VIRTUAL_ENV` before `srtctl apply` AND
   `mv .venv .venv-login` after apply.
4. **shared uv cache race**: 6 concurrent drivers corrupted `~/.cache/uv`
   ("failed to hardlink") -> per-job `UV_CACHE_DIR=/tmp/uvcache-<name>` + stagger launches.
5. **/mnt/home not mounted in container**: added `/mnt/home` + `/mnt/vast` to the
   driver's srtslurm.yaml `default_mounts` so the shim reads dataset + writes
   results to /mnt/home/.../results/<name> directly (live).
6. **sglang prefill OOM (mem_fraction_static=0.9)**: the template's 0.9 fits the
   stock 8k1k bench but NOT our agentic replay (longer/bursty) -> prefill OOM ~16min
   in -> decode KVTransferError. Fixed: `mem-fraction-static: 0.8` in all 3 sglang
   recipes (prefill+decode). Did NOT add expandable_segments (can conflict w/ mooncake cuMem).

### VERDICT (updated 2026-06-08 ~01:32) — 4 of 6 disagg jobs RUNNING
- **vLLM venv fix WORKED**: all 3 vllm jobs submitted+running — 6786 dsv4_vllm_4p_4d,
  6789 dsv4_vllm_8p_4d, 6792 dsv4_vllm_4p_8d (no more Exec format error).
- **sglang memfrac=0.8 fix holding** for 6777 dsv4_sglang_4p_4d + 6783 dsv4_sglang_4p_8d
  (running, no OOM as of init/early-sweep — still confirm they pass ~16min and write conc4.json).
- **BOTH multi-node sglang topologies FAIL at init (6780 8p_4d AND 6783 4p_8d)**:
  `RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)` at ~4min
  INIT (not sweep; memfrac=0.8 was applied). 8p_4d = 2-node prefill, 4p_8d = 2-node
  decode — both use `enable-dp-attention` (+ `megamoe` for 8p_4d) from the mid-curve
  templates. The single-node-each 4p_4d (plain TP4/TP4) runs FINE. NOTE: vLLM's
  multi-node topologies (8p_4d, 4p_8d) DO run — so this is a **sglang multi-node-role
  init bug**, not general. NEXT FIX to try: in the two sglang multi-node recipes drop
  `enable-dp-attention`/`megamoe` and use plain TP across the role's GPUs (e.g. TP8 for
  the 2-node role), or consult a known-good multi-node sglang dp-attention config.
  Recipes: sglang/.../agentic/disagg-gb300-{2p1d-tp4-tp4,1p2d-dep4-dep8}-agentic.yaml.
- So coverage = **4/6** RUNNING: 3 vllm (6786 4p_4d, 6789 8p_4d, 6792 4p_8d) + sglang
  4p_4d (6777). OPEN: both multi-node sglang topologies (8p_4d, 4p_8d).
- On resume: check `squeue`; for each running job `ls results/<name>/` for conc4/8/16.json
  + pareto.csv (+ server_metrics.prom). Disagg jobs are SLOW like colocated (~hrs).

## Why dsv4 (esp. sglang) "takes so long"
Not a bug — it's the workload x model. dsv4 FP4 is a huge MoE with expensive prefill
(~5-10k-token prefill batches) and low decode throughput (~200-420 tok/s) vs kimi
(~300-660). Our config = 60 sessions x (warmup + measured) x 3 conc points = **6 full
passes of 940k output tokens**. At dsv4's throughput that's ~5h/job colocated. Disagg
adds: ~12-20min setup per job (clone+srtctl+dynamo-wheel-build+multi-node model load)
+ KV-transfer/coordination overhead. kimi finishes ~3h. To speed up later: drop warmup
(2x), fewer conc points, or fewer sessions.

## Open follow-ups
- Confirm the disagg relaunch verdict; relaunch any failed job (fixes already applied).
- Collect/compare: 4 colocated pareto.csv + up to 6 disagg pareto.csv (+ server_metrics.prom).
- Wire real cache_hit_rate from server_metrics.prom into the pareto (was 0.0 client-side).
- DISAGG metrics: shim scrapes the Dynamo frontend /metrics (localhost:8000) every 15s ->
  results/<name>/server_metrics.prom. RICH set: dynamo_frontend_{time_to_first_token,
  inter_token_latency,request_duration,output_tokens_total,cached_tokens,input/output_sequence_tokens,
  total_kv_blocks,...} + dynamo_request_plane_{roundtrip_ttft,queue}_seconds (P/D routing).
  NOTE: this is FRONTEND-level (end-to-end+routing), NOT raw per-worker vllm:* counters —
  dynamo-vllm workers don't expose standalone /metrics on probed ports; dynamo aggregates.
  cached_tokens here gives the prefix-cache visibility the colocated client lacked.
- Old colocated confounded run preserved at results/{dsv4,kimi}_b1_repeats_prewarmup.
# ========== END HANDOFF ==========
---

## Goal
Replay our multi-turn **Claude Code** traces (KernelBook→Triton sessions) against a
real vLLM server and **sweep "number of concurrent sessions"** to capture
latency/throughput **pareto frontiers** for two models on one GB300 node (TP=4):
- **DeepSeek-V4 fp4** — `/mnt/vast/tom.chen/dsv4` (64 shards, `deepseek_v4`)
- **Kimi-K2.5 nvfp4** — `/mnt/vast/models/kimi-k25-nvfp4` (119 shards, `kimi_k25`)
  (user wanted "K2.6" but no K2.6 checkpoint exists on the cluster; using K2.5)

RULE: **never report fabricated/mock numbers** — only real vLLM (or sglang) runs.

## RIGHT NOW (jobs in flight — submitted 2026-06-07 ~23:04)
Four jobs, one model×engine each, on 4 separate gb300 nodes (181/183/185/187):
- **6717 vllm-dsv4**  -> `results/dsv4_b1`        (vllm 0.21.0 container)
- **6718 vllm-kimi**  -> `results/kimi_b1`        (vllm 0.21.0 container)
- **6719 sglang-dsv4**-> `results/dsv4_b1_sglang` (sglang deepseek-v4-grace-blackwell STABLE)
- **6720 sglang-kimi**-> `results/kimi_b1_sglang` (sglang nightly-cu13-20260602)
- Config (ALL): **60-session** dataset `replay/batch_3x.replay.jsonl`, **WARMUP=1**,
  **no --repeats** (flag removed), `CONCURRENCIES=4,8,16`. 12h `--time` cap.
- vllm jobs = proven recipe (low risk). **sglang jobs are UNPROVEN**: sglang must
  recognize `deepseek_v4` (dsv4) and load `kimi_k25`+NVFP4 (kimi) — watch their
  server.log for weight-load failures. sglang launchers omit tool/reasoning parsers
  for kimi (replayer discards output, so parsers are unneeded = smaller fail surface).

### What changed this round (DESIGN UPDATE)
1. **Removed `--repeats`** from replay_bench.py + sweep_pareto.py + launchers (it was
   confusing; re-replaying identical sessions = free cache hits).
2. **Added `--warmup`** to replay_bench.py / sweep_pareto.py: one UNTIMED full pass
   over the dataset before each concurrency point's measured pass, priming the prefix
   cache so every point starts warm. FIXES the dsv4 "TTFT falls as concurrency rises"
   artifact (cold-start outliers had all landed in conc4 = first sweep point).
   Default ON via `WARMUP=1` env in launchers.
3. **60-session dataset** `replay/batch_3x.replay.jsonl` (built by `replay/make_variants.py`):
   the 20 originals + 2 deterministically-refactored variants each (renamed kernel
   class + remapped problem-id so messages[1+] diverge early -> cache-distinct; turns
   preserved verbatim -> clean 3x load scale). 696 turns. Validated + dry-run OK.
4. New sglang launchers in fork: `benchmarks/single_node/agentic/{dsv4_fp4_b300_sglang_replay.sh,
   kimik2.5_fp4_b300_sglang_replay.sh}`. New sbatch wrappers `run_{dsv4,kimi}_sglang_replay.sbatch`.
5. Old confounded (repeats, no-warmup) vllm runs preserved at
   `results/{dsv4,kimi}_b1_repeats_prewarmup`.

### Resume / check status
```bash
squeue -u ppbhatt500                      # empty = all finished
sacct -j 6717,6718,6719,6720 --format=JobID,JobName,State,Elapsed,ExitCode
for d in dsv4_b1 kimi_b1 dsv4_b1_sglang kimi_b1_sglang; do
  echo "== $d =="; cat ~/gpumode-triton/results/$d/pareto.csv 2>/dev/null
done
# if a run isn't done: tail its server.log / slurm-<jobid>.out in that dir
```
When done: present ALL FOUR `pareto.csv` tables (vllm vs sglang × dsv4 vs kimi),
concurrency vs throughput/ttft/tpot/e2e + caveats. Numbers from files, not estimated.

## Key paths
- Dataset (model-agnostic): `~/gpumode-triton/replay/batch_1.replay.jsonl` (20 sessions, 232 turns)
- Canonical replay scripts: `~/gpumode-triton/replay/` (make_replay_dataset, replay_bench, sweep_pareto, analyze_replay)
- Fork: `~/InferenceX` (branch `custom-replay`)
  - launchers: `benchmarks/single_node/agentic/{dsv4_fp4_vllm_replay.sh, kimik2.5_fp4_b300_replay.sh}`
  - replay copies the launchers run: `utils/custom_replay/` (keep in sync with `~/gpumode-triton/replay/`)
- sbatch wrappers: `~/gpumode-triton/run_{kimi,dsv4}_replay.sbatch`
- Result dirs: `~/gpumode-triton/results/{kimi_b1,dsv4_b1}/` (each has a `RUN.md`)

## Serving environment (how it actually runs)
- vLLM isn't on the bare node — jobs run **inside the vLLM container** via Slurm pyxis/enroot:
  `--container-image=/mnt/vast/squash_dupe/vllm_vllm-openai_v0.21.0-ubuntu2404_arm64.sqsh`
  `--container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast`
  `--container-workdir=/mnt/home/ppbhatt500/InferenceX`
- That image was owner-only-read (bryan); **bryan chmod'd it readable** — if it reverts,
  re-running fails at "Invalid image format" (really a perms error).
- vLLM 0.21.0 DOES load `kimi_k25` + `deepseek_v4` with the `kimi_k2`/`deepseek_v4` parsers.

## Methodology (and the bug we fixed)
- **Completion-based** replay: each concurrency point replays ALL sessions' turns to
  completion; every turn is counted; run ends when all are done. `--concurrency` = #
  concurrent sessions (worker pool draining a shared queue; sequential within a session,
  parallel across workers).
- WHY: the earlier version measured for a fixed **60 s window** and only counted turns that
  finished in-window. dsv4 requests take 17–75 s, so most got **censored** → the dsv4
  TTFT-vs-concurrency curve was non-monotonic/garbage (looked like TTFT dropped at conc16
  because only the fast cache-hit requests survived the cutoff). Completion-based fixes this.
- `--repeats N` (default 1): replays the dataset N times. repeat>0 **prepends
  `"[session batch k] "` to `messages[0]`** so each extra pass is a realistic cold-cache
  MISS (not a free replay hit). At `repeats=1` nothing is appended — dataset is verbatim.
  Use repeats>1 to grow samples / push concurrency above the 20-session pool.

## Known caveats (real, seen in the prior 60s run; expect similar)
1. **Kimi drops some turns**: `failed_turns>0`, error `"no tokens streamed"` (dsv4 had 0).
   Likely a `kimi_k2` reasoning/tool-parser or stream-interval interaction. Report the count;
   investigate before fully trusting kimi per-turn aggregates.
2. **`cache_hit_rate` reads 0.0 in the CSV** even though the server IS doing ~85% prefix-cache
   reuse (see `kimi_b1/server.log`: "Prefix cache hit rate: 84–85%"). This vLLM build doesn't
   surface `prompt_tokens_details.cached_tokens` in the chat response, so the client can't see
   it. To put cache in the pareto, scrape the server `/metrics` endpoint in `sweep_pareto.py`.
3. dsv4 has very high TTFT (~11–15 s) even at low concurrency — real (big prefill), not a bug.

## Likely next steps (pick up here)
- Present the completed pareto tables for both models (the immediate deliverable).
- Optional: add low-concurrency points (1,2) for the latency floor — slow for dsv4 (~hrs at c1).
- Optional: wire `/metrics` cache scrape so `cache_hit_rate` is real in the pareto.
- Optional: chase the kimi `"no tokens streamed"` failures.
- Optional: wider sweep (1,2,4,8,16,32,64) with `--repeats` to push concurrency past 20.
- Not yet done: HF upload of the replay dataset; git-commit the latest launcher/script edits to the fork.
- Disagg + prom-stats frontend are stretch goals (not started).
