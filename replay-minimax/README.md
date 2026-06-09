# replay-minimax — simple, first-principles serving benchmark (MiniMax M2.5)

A deliberately small harness to understand **MiniMax-M2.5** serving from the only
tradeoff that matters: **throughput vs. interactivity / TTFT / E2E latency**.

We drive the server with real multi-turn agentic traces (Claude Code sessions —
model-agnostic: we replay the recorded prompts and force the recorded output
length with `ignore_eos` + `max_tokens`, so the decode load is reproduced on any
model) and read **Prometheus server metrics as ground truth** (never the client's
own timing math).

## The one graph that explains everything: throughput vs interactivity

For a fixed deployment, as you raise **concurrency** (number of simultaneous
users):

- **Throughput** (output tok/s **per GPU**) goes **up** — bigger decode batches
  use the GPU more efficiently. This is what you pay for.
- **Interactivity** (tok/s **per user** = `1000 / TPOT_ms`) goes **down** — each
  user's tokens are interleaved with more others, so they arrive slower.
- **TTFT** and **E2E** go **up** — prefill queues, requests wait.

The **pareto frontier** (throughput on Y, interactivity or TTFT/E2E on X) is the
deployment's identity: "how much throughput can I buy at a given latency SLA?"
Everything below (cache, offloading, disagg) is about **moving that frontier**.

We report the **p99 tail**, not p50 — tail latency is the SLA that matters.

## Metrics we collect (all from the server's `/metrics`, = ground truth)

Derived from first principles:

| Metric | Definition | Why it matters |
|---|---|---|
| throughput/GPU | `generation_tokens_total` Δ / runtime / n_gpu | GPU efficiency |
| interactivity | `1000 / inter_token_latency` (p50 & p99) | per-user speed |
| TTFT | `time_to_first_token_seconds` (p50/p99) | responsiveness |
| E2E | `e2e_request_latency_seconds` (p50/p99) | full request time |
| **GPU prefix-cache hit rate** | `gpu_prefix_cache_hit_rate` (or hits/queries Δ) | KV reuse — the biggest lever for multi-turn agentic |
| **GPU KV-cache usage** | `gpu_cache_usage_perc` | how close to capacity (preemption risk) |
| **CPU KV-cache usage** | `cpu_cache_usage_perc` / swap | only meaningful with offloading (stage 3) |
| queue depth | `num_requests_running` / `_waiting` | is the bottleneck prefill or decode? |

## Realistic load (carried over from the dsv4 work — see feedback below)

Two things make the measurement honest instead of a synthetic burst:

1. **Steady-state pre-arrange** (`--prearrange-frac 0.75`): a real server always
   has users *mid-conversation*. Before profiling, we drop each of the N
   concurrent users at a **random point ≤75%** through its own trace, using
   **one** warmup request each (that single request prefills the whole history,
   so the prefix lands **warm** in the cache — cheap: ~N requests, not a full
   replay). Then we start the stopwatch and resume each user from there. So at
   t=0 the users are at **mixed phases with warm caches**, not all at message #1.
2. **Staggered ramp** (`--ramp-seconds`) + **shuffled arrivals**: workers start
   spread over a window so they don't all hit prefill in the same instant
   (thundering herd → prefill spike, decode lockstep, simultaneous finish).

**Prefix caching is ON** (the existing MiniMax recipes ship it OFF — correct for
their random 1k1k benchmark, wrong for shared-prefix agentic traffic).

## Plan (stages — 1–3 are the priority)

1. **Aggregated (colocated) config.** One node, `vllm serve` MiniMax-M2.5,
   prefill+decode together, prefix caching on. Sweep concurrency → the
   throughput-vs-interactivity/TTFT/E2E pareto. *(serve_agg.sh + sweep.py)*
2. **Collect metrics.** GPU prefix-cache hit rate, KV-cache usage, queue depth —
   read from `/metrics` over the steady-state window. *(metrics.py, folded into
   the sweep output)*
3. **CPU offloading.** Re-run stage 1–2 with weight/KV offload to host RAM
   (`OFFLOAD_GB`), measure the **capacity-vs-latency** tradeoff (does it let us
   fit more KV / fewer GPUs, and what does it cost in TTFT/TPOT?).
4. **Disaggregated.** Split prefill/decode (Dynamo recipes) and repeat 1–3.
   *(added later)*

## Two metric sources (both collected)

1. **Prometheus serving metrics** (`/metrics`) = ground truth for throughput,
   latency, interactivity, **prefix-cache hit rate**, KV usage. (`metrics.py`)
2. **GPU hardware telemetry** via `nvidia-smi` sampled at 1 Hz across the whole
   sweep → power / utilization / temperature / memory, per-GPU and
   deployment-wide → `gpu.json`. (`gpu_metrics.py`) This is the physical side:
   throughput-per-Watt and GPU utilization tell you compute- vs memory-bound.

## Results

Everything is written under a dedicated **`../results_minimax/`**, one subdir per
config: `agg_vllm_tp4/`, `agg_vllm_tp4_offload40/`, `agg_sglang_tp4/`, … each with
`conc<N>.json` (per point), `pareto.csv`, `pareto.png`, and `gpu.json`.

## Files

- `serve_agg.sh`        — **vLLM** colocated serve + sweep (prefix caching ON,
  fp8 KV; `OFFLOAD_GB>0` → CPU offload). Runs the GPU monitor around the sweep.
- `serve_agg_sglang.sh` — **SGLang** colocated serve + sweep. *Identical flow*
  (same 150 sessions, pre-arrange 75%, ramp, prom + GPU metrics) — the **only**
  difference is the server process. RadixAttention prefix cache ON by default.
- `run_agg.sbatch` / `run_agg_sglang.sbatch` — Slurm wrappers (1 GB300 node +
  vLLM / SGLang container). Run on **separate nodes**; process is otherwise identical.
- `replay_bench.py` — one measured run: pre-arrange, staggered ramp, per-turn
  TTFT/ITL/E2E, brackets the window with a `/metrics` scrape → client +
  **server-truth** metrics.
- `sweep.py`        — sweep concurrencies → `pareto.csv` + `pareto.png`
  (throughput vs interactivity, vs TTFT, vs E2E; p99) + cache/KV per point.
- `metrics.py`      — Prometheus extraction; **engine-agnostic** (handles both
  `vllm:` and `sglang:` metric names — same file for both servers).
- `gpu_metrics.py`  — reduce the `nvidia-smi` CSV → per-GPU + deployment power/util/temp/mem.
- `scrape_metrics.py` — background `/metrics` poller → timestamped snapshots.

## Run

`batch_long.replay.jsonl` (the 150-session set) is already staged in this folder.

```bash
# vLLM, one node (stage 1+2):
MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 TP=4 CONCURRENCIES=4,8,16,32,64,128 \
  sbatch run_agg.sbatch
# stage 3 (CPU offload): add OFFLOAD_GB=40

# SGLang, a DIFFERENT node — identical process, only the server changes:
MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 TP=4 CONCURRENCIES=4,8,16,32,64,128 \
  sbatch run_agg_sglang.sbatch
```

**Blocker:** MiniMax-M2.5 is **not on `/mnt/vast/models` yet** — set `MODEL` to its
real path (or fetch `MiniMaxAI/MiniMax-M2.5`) before launching. `IMAGE=` overrides
the container (vLLM defaults to `v0.22.1-aarch64`; SGLang to the cluster nightly squash).
