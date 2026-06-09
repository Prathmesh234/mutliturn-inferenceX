#!/usr/bin/env python3
"""
Pre-canned multi-turn trace replayer for InferenceX-style pareto benchmarking.

Drives a vLLM (OpenAI-compatible) server with N CONCURRENT agentic sessions.
Each session replays its turns SEQUENTIALLY: turn k sends messages[:prefix_len_k]
and asks the server to generate exactly `max_tokens_k` tokens (ignore_eos), which
reproduces the recorded decode load. The server's output is timed but DISCARDED
("pre-canned" replay) -- the recorded assistant/user turns already sit in the
session's `messages`, so every turn's prompt is a deterministic, ever-growing
prefix of the last -> realistic KV prefix-cache reuse, identical across runs and
concurrency levels.

Completion-based: up to `--concurrency` sessions are in flight at once; each
worker replays one session's turns to completion then pulls the next, until ALL
sessions are done. No time window — every turn runs to completion and is counted
(nothing censored). `--concurrency` is the "number of concurrent sessions" knob
you sweep to trace the latency/throughput pareto.

`--warmup` runs one untimed pass over the whole dataset first, priming the server
prefix cache so every concurrency point is measured from the same warm state.

Metrics (per completed turn, server-authoritative token counts via
stream_options.include_usage): TTFT, TPOT, ITL, end-to-end latency, ISL, OSL,
cached prefix tokens. Aggregates are written as JSON compatible with InferenceX
result tooling.

Usage:
  python replay_bench.py --dataset replay/batch_1.replay.jsonl \
      --base-url http://0.0.0.0:8888 --model deepseek-ai/DeepSeek-V4-Pro \
      --concurrency 16 --warmup \
      --result-dir results --result-filename conc16.json

  python replay_bench.py --dataset replay/batch_1.replay.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_sessions(path: str) -> list[dict]:
    sessions = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            sessions.append(json.loads(line))
    return sessions


class TurnResult:
    __slots__ = ("ok", "ttft", "latency", "itls", "isl", "osl",
                 "cached", "finish_t", "error")

    def __init__(self):
        self.ok = False
        self.ttft = 0.0
        self.latency = 0.0
        self.itls: list[float] = []
        self.isl = 0
        self.osl = 0
        self.cached = 0
        self.finish_t = 0.0
        self.error = None


async def replay_turn(session_http, url, model, messages, max_tokens,
                      extra_body, timeout) -> TurnResult:
    """One streaming chat completion; times TTFT/ITL/e2e, reads server usage."""
    r = TurnResult()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    payload.update(extra_body)
    t0 = time.perf_counter()
    last = t0
    got_first = False
    try:
        async with session_http.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                r.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return r
            async for raw in resp.content:
                if not raw:
                    continue
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    # count any generated text/reasoning token arrival for ITL
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if not got_first:
                            r.ttft = now - t0
                            got_first = True
                        else:
                            r.itls.append(now - last)
                        last = now
                usage = chunk.get("usage")
                if usage:
                    r.isl = int(usage.get("prompt_tokens", 0) or 0)
                    r.osl = int(usage.get("completion_tokens", 0) or 0)
                    details = usage.get("prompt_tokens_details") or {}
                    r.cached = int(details.get("cached_tokens", 0) or 0)
        r.latency = time.perf_counter() - t0
        r.finish_t = time.perf_counter()
        r.ok = got_first
        if not got_first and r.error is None:
            r.error = "no tokens streamed"
    except Exception as e:  # noqa: BLE001
        r.error = f"{type(e).__name__}: {e}"
    return r


def _parse_prom(text: str) -> dict:
    """Parse Prometheus text exposition into {series_line: value}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sp = line.rsplit(" ", 1)   # `name{labels} value`
        if len(sp) != 2:
            continue
        try:
            out[sp[0]] = float(sp[1])
        except ValueError:
            continue
    return out


async def _scrape_prom(http, url: str):
    """GET the server /metrics endpoint and parse it (best-effort, never fatal)."""
    try:
        async with http.get(url) as resp:
            return _parse_prom(await resp.text())
    except Exception as e:  # noqa: BLE001
        print(f"[replay] prom scrape failed ({url}): {e}", flush=True)
        return None


def _write_prom(args, before, after):
    """Write c<conc>_prom.json: before/after server counters + their delta."""
    try:
        rd = Path(args.result_dir); rd.mkdir(parents=True, exist_ok=True)
        delta = {}
        if before and after:
            for k, v in after.items():
                if k in before and (v - before[k]) != 0:
                    delta[k] = round(v - before[k], 6)
        out = rd / f"c{args.concurrency}_prom.json"
        out.write_text(json.dumps({
            "concurrency": args.concurrency,
            "prom_url": args.prom_url,
            "n_series": len(after) if after else 0,
            "before": before or {},
            "after": after or {},
            "delta": delta,
        }, indent=2))
        print(f"[replay] wrote {out} ({len(after or {})} series, "
              f"{len(delta)} changed)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[replay] prom write skipped: {e}", flush=True)


async def run_benchmark(args):
    import aiohttp

    sessions = load_sessions(args.dataset)
    if not sessions:
        print("no sessions in dataset", file=sys.stderr)
        sys.exit(1)
    url = args.base_url.rstrip("/") + args.endpoint
    extra_body = json.loads(args.extra_body) if args.extra_body else {}

    # Completion-based: every session is replayed to completion exactly once and
    # EVERY turn is recorded. No time window, no censoring — the run ends only
    # when all turns have finished. Concurrency = how many sessions are in flight
    # at once: each worker pulls a session from the shared queue, replays its
    # turns sequentially to completion, then pulls the next. To grow the workload
    # past the recorded session count, refactor the dataset into more DISTINCT
    # sessions (see replay/make_variants.py) instead of re-replaying identical
    # ones — identical replays would just be free prefix-cache hits.
    work = list(sessions)
    if args.shuffle_sessions:
        # Randomize session order (seeded for reproducibility) so the staggered
        # arrivals bring in a realistic RANDOM mix of session sizes. Better than a
        # length sort, which would correlate size with arrival time (heavy prefills
        # bunching at the tail = a second mini-herd). Combined with --ramp-seconds
        # this removes the synchronized turn-0 prefill burst (thundering herd).
        import random
        random.Random(args.seed).shuffle(work)

    # Pre-arrange (steady-state snapshot): position the first `concurrency` sessions
    # at a random point up to --prearrange-frac through their conversation. The setup
    # phase sends ONE request per session (turn k_i) to warm its prefix cache; the
    # measured phase then resumes each from turn k_i+1. So at t0 the in-flight users
    # are at MIXED phases with WARM caches — like a real server snapshot — instead of
    # all starting fresh at turn 0 (thundering herd). Sessions pulled later (fresh
    # arrivals) start at turn 0.
    if args.prearrange_frac > 0:
        import random as _r
        _rng = _r.Random(args.seed + 1)
        for s in work[: min(args.concurrency, len(work))]:
            nt = len(s["turns"])
            if nt >= 2:  # need a turn k_i to warm AND at least one turn to profile
                k = min(int(_rng.uniform(0, args.prearrange_frac) * nt), nt - 2)
                s["_warm_idx"] = k        # the single warmup request (turn k_i)
                s["_measure_from"] = k + 1  # profiling resumes here

    total_turns = sum(len(s["turns"]) for s in work)
    results: list[TurnResult] = []

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    n_workers = min(args.concurrency, len(work))
    if args.concurrency > len(work):
        print(f"[replay] concurrency {args.concurrency} > {len(work)} sessions; "
              f"effective concurrency capped at {len(work)} (add more distinct "
              f"sessions to the dataset)", flush=True)
    print(f"[replay] concurrency {args.concurrency}: replaying {len(work)} sessions / "
          f"{total_turns} turns to completion (no time limit)"
          + (" [+warmup]" if args.warmup else ""), flush=True)

    async with aiohttp.ClientSession(connector=conn, headers=headers) as http:

        async def drain(record: bool, limit: int):
            counter = {"i": 0}
            nw = min(n_workers, limit)

            async def worker(wid: int):
                # Staggered ramp: spread the workers' first-turn starts evenly across
                # --ramp-seconds so they don't all fire prefill at the same instant
                # (thundering herd). Only on the measured pass.
                if record and args.ramp_seconds and nw > 1:
                    await asyncio.sleep(args.ramp_seconds * wid / nw)
                while True:
                    i = counter["i"]
                    counter["i"] += 1
                    if i >= limit:
                        return
                    s = work[i]
                    msgs = s["messages"]
                    # Pre-arranged sessions resume from turn k_i+1 (prefix already
                    # warmed in the pre-arrange phase); fresh sessions start at 0.
                    for turn in s["turns"][s.get("_measure_from", 0):]:
                        if args.use_think_time and turn.get("delay_before_s"):
                            await asyncio.sleep(min(turn["delay_before_s"], args.max_think_time))
                        tr = await replay_turn(
                            http, url, args.model, msgs[:turn["prefix_len"]],
                            turn["max_tokens"], extra_body, timeout)
                        if record:
                            results.append(tr)   # record EVERY turn — nothing censored

            await asyncio.gather(*[worker(k) for k in range(nw)])

        async def prearrange(limit: int):
            # ONE untimed request per initial session, at its random mid-point k_i,
            # to warm that session's prefix cache. All workers run concurrently; we
            # await all before profiling so every in-flight user is positioned.
            counter = {"i": 0}

            async def pworker():
                while True:
                    i = counter["i"]
                    counter["i"] += 1
                    if i >= limit:
                        return
                    s = work[i]
                    wi = s.get("_warm_idx")
                    if wi is None:
                        continue
                    turn = s["turns"][wi]
                    await replay_turn(http, url, args.model,
                                      s["messages"][:turn["prefix_len"]],
                                      turn["max_tokens"], extra_body, timeout)  # not recorded

            await asyncio.gather(*[pworker() for _ in range(min(n_workers, limit))])

        # Setup phase. Pre-arrange (steady-state snapshot: 1 request per initial
        # session to warm it mid-conversation) takes precedence; else the legacy
        # full-prefix warmup.
        if args.prearrange_frac > 0:
            nlim = min(args.concurrency, len(work))
            npre = sum(1 for s in work[:nlim] if "_warm_idx" in s)
            print(f"[replay] pre-arrange: positioning {npre}/{nlim} initial sessions "
                  f"<= {args.prearrange_frac:.0%} through (1 warmup request each)...", flush=True)
            await prearrange(nlim)
            print("[replay] pre-arrange done; profiling resumes each session from k_i+1",
                  flush=True)
        elif args.warmup:
            wlim = args.warmup_sessions if args.warmup_sessions and args.warmup_sessions > 0 else len(work)
            wlim = min(wlim, len(work))
            print(f"[replay] warmup: untimed pass over {wlim}/{len(work)} sessions "
                  f"to prime the prefix cache...", flush=True)
            await drain(record=False, limit=wlim)
            print("[replay] warmup done; starting measured pass", flush=True)

        # Snapshot server-side Prometheus counters bracketing the measured pass.
        prom_before = await _scrape_prom(http, args.prom_url) if args.prom_url else None

        t0 = time.perf_counter()
        await drain(record=True, limit=len(work))
        runtime = time.perf_counter() - t0

        if args.prom_url:
            prom_after = await _scrape_prom(http, args.prom_url)
            _write_prom(args, prom_before, prom_after)

    print(f"[replay] done: {len(results)}/{total_turns} turns in {runtime:.1f}s "
          f"at concurrency {args.concurrency}"
          + (" (after warmup)" if args.warmup else ""), flush=True)
    return aggregate(results, args, runtime)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def aggregate(results: list[TurnResult], args, runtime: float) -> dict:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    window = runtime if runtime > 0 else 1e-9   # real wall-clock to completion
    ttft = np.array([r.ttft * 1000 for r in ok])           # ms
    e2e = np.array([r.latency * 1000 for r in ok])         # ms
    # TPOT = mean inter-token latency per request (ms/token), excludes TTFT
    tpot = np.array([np.mean(r.itls) * 1000 for r in ok if r.itls])
    itl_all = np.array([x * 1000 for r in ok for x in r.itls])
    isl = np.array([r.isl for r in ok])
    osl = np.array([r.osl for r in ok])
    cached = np.array([r.cached for r in ok])
    tot_out = int(osl.sum())
    tot_in = int(isl.sum())
    summary = {
        "concurrency": args.concurrency,
        "runtime_s": round(window, 2),
        "completed_turns": len(ok),
        "failed_turns": failed,
        # throughput
        "request_throughput_per_s": round(len(ok) / window, 4),
        "output_throughput_tok_per_s": round(tot_out / window, 2),
        "total_token_throughput_tok_per_s": round((tot_in + tot_out) / window, 2),
        # latency (ms)
        "ttft_ms": {"mean": round(float(ttft.mean()) if len(ttft) else 0, 2),
                    "p50": round(_pct(ttft, 50), 2), "p90": round(_pct(ttft, 90), 2),
                    "p95": round(_pct(ttft, 95), 2), "p99": round(_pct(ttft, 99), 2)},
        "tpot_ms": {"mean": round(float(tpot.mean()) if len(tpot) else 0, 2),
                    "p50": round(_pct(tpot, 50), 2), "p90": round(_pct(tpot, 90), 2),
                    "p95": round(_pct(tpot, 95), 2), "p99": round(_pct(tpot, 99), 2)},
        "itl_ms": {"mean": round(float(itl_all.mean()) if len(itl_all) else 0, 2),
                   "p99": round(_pct(itl_all, 99), 2)},
        "e2e_ms": {"mean": round(float(e2e.mean()) if len(e2e) else 0, 2),
                   "p50": round(_pct(e2e, 50), 2), "p90": round(_pct(e2e, 90), 2),
                   "p95": round(_pct(e2e, 95), 2), "p99": round(_pct(e2e, 99), 2)},
        # workload shape (server-authoritative)
        "isl": {"mean": round(float(isl.mean()) if len(isl) else 0, 1),
                "median": int(np.median(isl)) if len(isl) else 0,
                "p95": int(_pct(isl, 95))},
        "osl": {"mean": round(float(osl.mean()) if len(osl) else 0, 1),
                "median": int(np.median(osl)) if len(osl) else 0,
                "p95": int(_pct(osl, 95))},
        # cache: fraction of prompt tokens served from the prefix cache
        "cache_hit_rate": round(float(cached.sum() / tot_in), 4) if tot_in else 0.0,
        "cached_tokens_total": int(cached.sum()),
        "input_tokens_total": tot_in,
        "output_tokens_total": tot_out,
        "errors_sample": [r.error for r in results if r.error][:5],
    }

    # raw per-request arrays for measured bootstrap CIs (additive; never fatal)
    try:
        _rd = Path(args.result_dir); _rd.mkdir(parents=True, exist_ok=True)
        _tmp = _rd / f".raw_conc{args.concurrency}.jsonl.tmp"
        with _tmp.open("w") as _fh:
            for r in ok:
                _fh.write(json.dumps({
                    "ttft_ms": r.ttft * 1000,
                    "tpot_ms": (float(np.mean(r.itls)) * 1000) if r.itls else None,
                    "e2e_ms": r.latency * 1000,
                    "isl": r.isl, "osl": r.osl,
                }) + "\n")
        _tmp.replace(_rd / f"raw_conc{args.concurrency}.jsonl")
    except Exception as _e:
        print(f"[replay] raw dump skipped: {_e}", flush=True)
    return summary


def dry_run(args):
    sessions = load_sessions(args.dataset)
    turns = [t for s in sessions for t in s["turns"]]
    osl = [t["max_tokens"] for t in turns]
    print(f"[dry-run] {len(sessions)} sessions, {len(turns)} turns "
          f"({len(turns)/max(len(sessions),1):.1f}/session)")
    print(f"[dry-run] planned OSL (max_tokens): median {int(np.median(osl))} "
          f"max {max(osl)} sum {sum(osl):,}")
    print(f"[dry-run] would replay all {len(sessions)} sessions / "
          f"{len(turns)} turns to completion at "
          f"concurrency={args.concurrency} against {args.base_url}{args.endpoint}")
    print("[dry-run] OK — dataset is replay-ready.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="*.replay.jsonl from make_replay_dataset.py")
    ap.add_argument("--base-url", default="http://0.0.0.0:8888")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="number of concurrent sessions (the pareto sweep knob)")
    ap.add_argument("--warmup", action="store_true",
                    help="run one untimed full pass over the dataset before the "
                         "measured pass, to prime the server prefix cache so every "
                         "concurrency point starts from the same warm cache state.")
    ap.add_argument("--warmup-sessions", type=int, default=0,
                    help="if >0, the untimed warmup pass covers only the first N "
                         "sessions (a cheap prime) instead of the whole dataset; "
                         "the measured pass still covers all sessions.")
    ap.add_argument("--prom-url", default=None,
                    help="if set, scrape this Prometheus /metrics URL before and "
                         "after the measured pass and write c<conc>_prom.json "
                         "(server-side counters: prefix-cache hits, KV usage, etc.).")
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--use-think-time", action="store_true",
                    help="sleep the recorded inter-turn idle gap before each turn "
                         "(default off = saturate for throughput)")
    ap.add_argument("--max-think-time", type=float, default=60)
    ap.add_argument("--ramp-seconds", type=float, default=0.0,
                    help="stagger worker starts evenly across this window on the "
                         "measured pass to avoid a synchronized turn-0 prefill burst "
                         "(thundering herd). 0 = all workers start at once.")
    ap.add_argument("--shuffle-sessions", action="store_true",
                    help="randomize session order (seeded) so staggered arrivals "
                         "bring a realistic random size mix; avoids the t=0 herd and "
                         "any length/arrival-time correlation a sort would add.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for --shuffle-sessions / --prearrange-frac.")
    ap.add_argument("--prearrange-frac", type=float, default=0.0,
                    help="steady-state snapshot: position the first `concurrency` "
                         "sessions at a random point in [0, FRAC) of their conversation "
                         "via ONE warmup request each, then profile from k_i+1. "
                         "0 = off. e.g. 0.75 = up to 75%% through.")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged into every request body (e.g. sampling params)")
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--result-filename", default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate dataset, hit no server")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    summary = asyncio.run(run_benchmark(args))
    print(json.dumps(summary, indent=2))
    rd = Path(args.result_dir)
    rd.mkdir(parents=True, exist_ok=True)
    fn = args.result_filename or f"replay-conc{args.concurrency}.json"
    (rd / fn).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {rd / fn}")


if __name__ == "__main__":
    main()
