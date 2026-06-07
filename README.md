# gpumode-triton

Solve [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook)
problems by converting PyTorch modules to **Triton**, using `claude -p`
(Opus 4.8, `claude-opus-4-8`) in parallel across the **4 GPUs of one GB300 node**.

## How it works (simple version)

```
solve.py (1 process)
  ├─ load a batch of problems from KernelBook  → shared queue
  └─ spawn 4 workers, worker i pinned to GPU i
         loop: pull problem from queue
               write reference.py into the work dir
               claude -p (GPU i pinned, NO turn cap):
                   pipe kernel:  python judge.py <<'PYEOF' ... PYEOF
                   judge runs it on the GPU → correctness+speedup back →
                   fix/optimize → pipe again ...
                   (up to --max-evals judge runs = the iteration budget)
               authoritative re-score of saved submission.py on GPU i
               append 1 line to gpu_i.jsonl  (incl. full eval_trajectory)
               → next problem
```

- **One worker per GPU**, dynamic pull queue (a slow problem doesn't stall others).
- **Each worker writes its own `runs/gpu_<i>.jsonl`** — no contention.
- **Resumable**: re-running skips problems already present in `runs/gpu_*.jsonl`.
- `claude -p` runs against the **Anthropic API** (not the local GPU). The GPU is
  used only by `judge.py` to run the generated kernel.
- **One drop-in script, `judge.py`.** Claude hands it a kernel on **stdin** and
  gets back correctness + speedup. The judge saves the kernel to `submission.py`,
  counts each call, and logs every attempt.
- **No `--max-turns`.** Claude thinks/drafts freely; the budget is the number of
  **judge runs** via `--max-evals` (default 8). Each one = running the kernel on
  the GPU and getting feedback. The judge tells Claude to stop when the budget is
  spent. The hard safety backstop is the wall-clock `--claude-timeout`.

## Files

| File | Purpose |
|------|---------|
| `solve.py`        | orchestrator + per-GPU worker loop + queue |
| `claude_runner.py`| runs `claude -p`, captures the trace, **waits out rate limits** |
| `judge.py`        | the single drop-in evaluator: kernel on **stdin** → correctness + speedup; counts evals = the budget; `--score` re-scores for the final record |
| `run.sh`          | Slurm launcher (bootstraps arch-correct `uv`, installs deps, runs) |
| `pyproject.toml`  | `uv` project (core deps + optional `eval` extra) |

## Run it

```bash
cd ~/gpumode-triton

# one batch of 300 (problems 0..299)
sbatch run.sh

# next batch
START=300 BATCH=300 sbatch run.sh

# watch it
tail -f runs/slurm-*.out
squeue -u $USER
```

To run interactively on a node instead of via the batch queue:

```bash
srun --partition=gb300 --nodes=1 --gres=gpu:4 --cpus-per-task=64 \
     --time=02:00:00 --pty bash run.sh
```

## Cluster topology & rules

This is a Slurm cluster with a fixed shape we stay within:

| | |
|---|---|
| Racks | **2** — `DH1-153-US-EAST-04B`, `DH2-058-US-EAST-04B` (Slurm topology blocks) |
| Servers / rack | **18** |
| GPUs / server | **4** (`gpu:gb300:4`) |

This job uses **1 server / 4 GPUs**, comfortably within the rules. The 4 workers
share **one** `uv` venv (`.venv`) — read-only at runtime, so it's safe to share,
and since `/mnt/home` is shared and every node is `aarch64`, the same venv works
on any GPU of any server in either rack. GPU isolation is via `CUDA_VISIBLE_DEVICES`,
not separate environments.

**Scaling (still within the rules):** run **one job per server** — a Slurm array
of `--nodes=1 --gres=gpu:4` jobs, each with a different `START` offset — rather
than one giant job. If you ever do submit a multi-node job, keep it `<= 18` nodes
and pin it to a single rack with `--constraint=DH1-153-US-EAST-04B` (or `DH2-...`)
so it stays inside one NVLink/block domain. (Note: `solve.py` itself is
single-node; multi-server scaling is done by launching more jobs, not by widening
one job.)

## Result fidelity & GPU robustness

- **No result caching.** Triton's `TRITON_CACHE_DIR` caches *compiled binaries*, not
  output values — it can't make results stale. Every eval runs in its **own fresh
  process** (no autotune/allocator/warm-tensor carryover), each problem gets its own
  cache dir, and `triton.testing.do_bench` **flushes the L2 cache before every timed
  run** (verified in triton 3.6.0), so timings are uncached/"true". `--fresh-compile`
  additionally wipes the compile cache before each eval (forces cold compile; does
  not change results — default off).
- **GPU isolation = state reset.** Each eval is a separate process, so OOM / illegal
  access / crash dies with that process and the driver frees its memory; the next eval
  starts on a clean CUDA context. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  reduces fragmentation OOMs.
- **Cascade breaker.** After a GPU-suspicious failure (timeout/runtime_error/crash),
  the worker runs a tiny CUDA health check; if the GPU is wedged it cools down
  (`--gpu-cooldown`) and re-checks before continuing, and records `gpu_health`
  (`ok`/`recovered`/`unhealthy`). A truly wedged GPU (Xid) needs a root-level reset we
  can't do — this contains and surfaces it rather than failing every later problem.

## Output

`runs/gpu_<i>.jsonl`, one JSON object per solved problem:

```jsonc
{
  "uuid": "...", "entry_point": "SumAggregator",
  "claude_status": "ok",            // ok | rate_limited | error | timeout
  "num_evals": 3, "cost_usd": 0.04, "elapsed_s": 73.2,
  "eval": { "status": "ok", "correct": true,   // authoritative final score
            "speedup": 1.8, "ref_ms": 0.21, "new_ms": 0.12 },
  "gpu_health": "ok",               // ok | recovered | unhealthy
  "eval_trajectory": [              // one entry per check.py run Claude did
    { "attempt": 1, "status": "incorrect", "correct": false },
    { "attempt": 2, "status": "ok", "correct": true, "speedup": 1.3 },
    { "attempt": 3, "status": "ok", "correct": true, "speedup": 1.8 }
  ],
  "generated_code": "import triton...",
  "trace": [ /* full stream-json: every turn + tool call */ ]
}
```

Quick look at results:

```bash
cat runs/gpu_*.jsonl | jq -c '{uuid, claude_status, eval: .eval.status, speedup: .eval.speedup}'
# success rate
cat runs/gpu_*.jsonl | jq -r '.eval.status' | sort | uniq -c
```

## Rate-limit handling

In `claude_runner.py`: a failed `claude -p` whose output matches rate/usage-limit
patterns is treated as **rate-limited** — the worker sleeps (until the parsed
reset time if present, else exponential backoff `--rl-base`→`--rl-max` with
jitter) and **retries the same problem**. Non-rate-limit errors get a few quick
retries (`--max-retries`). Tunables are flags on `solve.py`.

> Auth is the **subscription/OAuth token**, which throttles hardest. Throughput
> is capped by that limit, not by the GPUs. If you later get an `ANTHROPIC_API_KEY`
> or Bedrock/Vertex access, you can raise `--num-workers` well above 4 and go faster.

## Before the first big run — two checks

1. **Auth works headless on a compute node** (uses a few cents):
   ```bash
   srun --partition=gb300 --gres=gpu:1 --time=5 --pty \
     bash -lc 'claude -p "reply OK" --max-turns 1 --output-format json'
   ```
2. **GPU env** builds via `uv` with a **managed** CPython 3.12 (NOT system python —
   it lacks the `Python.h` headers Triton needs to JIT-compile). `run.sh` does this
   for you; to verify on a node:
   ```bash
   srun --partition=gb300 --gres=gpu:1 --time=10 \
     bash -lc 'export PATH=$HOME/.local/uv-$(uname -m):$PATH; cd ~/gpumode-triton; \
               uv sync --python 3.12 --extra eval && uv run python verify_env.py'
   ```
   Expected: `device: NVIDIA GB300`, `capability (10, 3)`, `triton kernel correct: True`,
   `ENV OK`. (Verified stack: torch 2.11.0+cu128, triton 3.6.0.)
