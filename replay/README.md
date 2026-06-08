# Agentic trace replay — Claude Code → InferenceX pareto

Replay our multi-turn **Claude Code** traces (KernelBook→Triton sessions) against
an OpenAI-compatible inference server and sweep **number of concurrent sessions**
to trace the latency/throughput **pareto frontier**. Target: **DeepSeek-V4 FP4**
on vLLM, single node. No weka, no aiperf — a small, self-contained replayer.

Everything uses **uv** for deps (`uv run --with …`, no venv to manage).

## The pipeline (4 scripts)

| script | what it does |
|---|---|
| `make_replay_dataset.py` | `batch_N.json` (raw traces) → standardized `*.replay.jsonl` (flat OpenAI `messages` + per-turn `{prefix_len, max_tokens, delay}`) |
| `replay_bench.py` | drives the server: N concurrent sessions, pre-canned replay, captures TTFT/TPOT/ITL/ISL/OSL/cache |
| `sweep_pareto.py` | sweeps `--concurrency`, writes `pareto.csv` + `pareto.png` |
| `analyze_replay.py` | data analysis of the traces: depth, ISL/OSL, cache reuse, fan-out |

## Pre-canned replay (what "replay" means here)

A session is reconstructed once into a flat `messages` list. Turn *k* sends
`messages[:prefix_len_k]` and asks the server to generate exactly `max_tokens_k`
tokens (`ignore_eos`) — reproducing the recorded decode load — then **discards**
the output. The recorded assistant/user turns are already in `messages`, so each
turn's prompt is a deterministic, ever-growing prefix of the previous one →
realistic KV prefix-cache reuse, identical across runs and concurrency levels.

> OSL note: Claude's per-event `output_tokens` omit hidden thinking tokens, so we
> distribute the **authoritative session-total** output (from the trace's `result`
> event) across turns weighted by visible content. Per-session OSL sums match the
> trace exactly; `ignore_eos` then reproduces that decode load on any model.

> System prompt: `system_prompt.md` is prepended as message 0 of every session
> (identical across sessions → shared cached prefix). Disable with `--system-prompt ''`.

## Plug-and-play: drop a `batch_N.json`, get a pareto

```bash
cd ~/gpumode-triton
# 1. convert (any runs_*/batch_*.json — the format is identical)
uv run --with numpy python replay/make_replay_dataset.py \
    runs_8t/batch_1.json -o replay/batch_1.replay.jsonl

# 2. analyze the traces (offline, no server)
uv run --with numpy --with matplotlib python replay/analyze_replay.py \
    replay/batch_1.replay.jsonl --raw runs_8t/batch_1.json -o results/analysis

# 3. sweep concurrency against a running vLLM server -> pareto
uv run --with aiohttp --with numpy --with matplotlib python replay/sweep_pareto.py \
    --dataset replay/batch_1.replay.jsonl \
    --base-url http://0.0.0.0:8888 --model deepseek-ai/DeepSeek-V4-Pro \
    --concurrencies 1,2,4,8,16,32,64,128 --duration 120 --warmup 20 \
    --result-dir results/dsv4_b1
```

Single concurrency point / smoke test:

```bash
uv run --with aiohttp --with numpy python replay/replay_bench.py \
    --dataset replay/batch_1.replay.jsonl --dry-run            # validate, no server
uv run --with aiohttp --with numpy python replay/replay_bench.py \
    --dataset replay/batch_1.replay.jsonl --base-url http://0.0.0.0:8888 \
    --model deepseek-ai/DeepSeek-V4-Pro --concurrency 16 --duration 60 --warmup 10
```

The server itself (DeepSeek-V4 **FP4** on vLLM) is launched by the InferenceX
fork recipe `benchmarks/single_node/agentic/dsv4_fp4_vllm_replay.sh`, which also
runs the sweep end-to-end. See that script for the SLURM/one-node entrypoint.

## Standardized format (`*.replay.jsonl`, one session per line)

```json
{
  "session_id": "…", "entry_point": "MultiHeadAttention",
  "source": "GPUMODE/KernelBook", "kernelbook_uuid": 311,
  "messages": [{"role": "system|user|assistant", "content": "…"}, …],
  "turns": [{"prefix_len": 2, "max_tokens": 671, "delay_before_s": 0.0,
             "rec_input_tokens": 23522, "rec_output_tokens": 671,
             "rec_cache_read_tokens": 15825}, …],
  "recorded": {"num_turns": 13, "correct": true, "speedup": 1.72}
}
```

Upload to HF with `push_hf.py` (dataset repo of your choice) once you're happy
with a batch.
