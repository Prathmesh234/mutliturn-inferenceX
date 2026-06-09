#!/usr/bin/env python3
"""One measured benchmark run at a fixed concurrency against a colocated
OpenAI-compatible server (vLLM serving MiniMax-M2.5).

Load model (first principles):
  - N concurrent multi-turn sessions (real agentic traces).
  - STEADY-STATE pre-arrange: each of the first N sessions is positioned at a
    random point <= --prearrange-frac through its conversation, via ONE untimed
    warmup request (prefills + caches its history). Profiling then resumes each
    from k_i+1. => at t0 users are mid-conversation, mixed phases, WARM cache.
  - Staggered ramp + shuffle => no synchronized turn-0 prefill burst.

Metrics: client TTFT/ITL (sanity) + ACTUAL client E2E (exact per-turn latency, the
true end-to-end — the server prom e2e histogram is bucket-coarse above 60s) + SERVER
PROMETHEUS TRUTH (throughput, interactivity, cache hit rate, KV usage) via metrics.summarize.

Usage:
  python replay_bench.py --dataset batch_long.replay.jsonl --base-url http://localhost:8000 \
    --model minimax --concurrency 16 --prearrange-frac 0.75 --ramp-seconds 60 \
    --metrics-url http://localhost:8000/metrics --n-gpu 4 --result-dir out --result-filename c16.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import metrics as M


def load_sessions(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, int(q * len(xs)))
    return round(xs[i], 3)


async def fetch(http, url, timeout):
    import aiohttp
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            return await r.text()
    except Exception as e:  # noqa: BLE001
        print(f"[replay] /metrics scrape failed: {e}", file=sys.stderr)
        return ""


async def replay_turn(http, url, model, messages, max_tokens, extra_body, timeout):
    """Stream one turn; return (ttft_ms, itl_ms_list, e2e_ms, out_toks, ok)."""
    import aiohttp
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": True, "temperature": 0.0, **extra_body}
    t0 = time.perf_counter()
    ttft = None
    last = t0
    itls = []
    out = 0
    try:
        async with http.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                ch = obj.get("choices") or []
                delta = ch[0].get("delta", {}) if ch else {}
                tok = delta.get("content") or delta.get("reasoning_content")
                if tok:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = (now - t0) * 1000
                    else:
                        itls.append((now - last) * 1000)
                    last = now
                    out += 1
    except Exception:  # noqa: BLE001
        return None, [], None, 0, False
    if ttft is None:
        return None, [], None, 0, False
    e2e = (time.perf_counter() - t0) * 1000
    return ttft, itls, e2e, out, True


async def run(args):
    import aiohttp

    sessions = load_sessions(args.dataset)
    work = list(sessions)
    if args.shuffle_sessions:
        random.Random(args.seed).shuffle(work)

    # pre-arrange: position the first `concurrency` sessions mid-conversation
    if args.prearrange_frac > 0:
        rng = random.Random(args.seed + 1)
        for s in work[: min(args.concurrency, len(work))]:
            nt = len(s["turns"])
            if nt >= 2:
                # k_i ~ uniform in the [lo, hi] band (Cam: "middling quantiles 25-75%")
                frac = rng.uniform(args.prearrange_lo, args.prearrange_frac)
                k = min(max(int(frac * nt), 0), nt - 2)  # >=0, leave >=1 turn to profile
                s["_warm_idx"] = k
                s["_measure_from"] = k + 1

    url = args.base_url.rstrip("/") + args.endpoint
    extra = json.loads(args.extra_body) if args.extra_body else {}
    nw = min(args.concurrency, len(work))
    ttfts, e2es, all_itls, outs = [], [], [], []
    cached = {"ok": 0, "fail": 0}

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as http:

        async def prearrange():
            ctr = {"i": 0}

            async def w():
                while True:
                    i = ctr["i"]; ctr["i"] += 1
                    if i >= nw:
                        return
                    s = work[i]; wi = s.get("_warm_idx")
                    if wi is None:
                        continue
                    t = s["turns"][wi]
                    await replay_turn(http, url, args.model, s["messages"][:t["prefix_len"]],
                                      t["max_tokens"], extra, args.request_timeout)
            await asyncio.gather(*[w() for _ in range(nw)])

        async def measure():
            ctr = {"i": 0}

            async def w(wid):
                if args.ramp_seconds and nw > 1:
                    await asyncio.sleep(args.ramp_seconds * wid / nw)
                while True:
                    i = ctr["i"]; ctr["i"] += 1
                    if i >= len(work):
                        return
                    s = work[i]; msgs = s["messages"]
                    for t in s["turns"][s.get("_measure_from", 0):]:
                        ttft, itls, e2e, out, ok = await replay_turn(
                            http, url, args.model, msgs[:t["prefix_len"]],
                            t["max_tokens"], extra, args.request_timeout)
                        if ok:
                            cached["ok"] += 1
                            ttfts.append(ttft); e2es.append(e2e); all_itls.extend(itls); outs.append(out)
                        else:
                            cached["fail"] += 1
            await asyncio.gather(*[w(k) for k in range(nw)])

        # setup (untimed): position users mid-conversation, warm their caches
        if args.prearrange_frac > 0:
            npre = sum(1 for s in work[:nw] if "_warm_idx" in s)
            print(f"[replay] pre-arrange: positioning {npre}/{nw} sessions "
                  f"{args.prearrange_lo:.0%}-{args.prearrange_frac:.0%} through "
                  f"(1 warmup request each)...", flush=True)
            await prearrange()
            print("[replay] pre-arrange done; profiling resumes from k_i+1", flush=True)

        # bracket the measured window with a /metrics scrape (server = ground truth)
        prom_before = await fetch(http, args.metrics_url, 15) if args.metrics_url else ""
        t0 = time.perf_counter()
        await measure()
        runtime = time.perf_counter() - t0
        prom_after = await fetch(http, args.metrics_url, 15) if args.metrics_url else ""

    # Persist raw before/after /metrics snapshots so any derived metric (cache hit
    # rate, KV usage, ...) can be recomputed later even if a metric name drifts.
    if (prom_before or prom_after):
        stem = (args.result_filename or f"conc{args.concurrency}.json").rsplit(".", 1)[0]
        rd = Path(args.result_dir); rd.mkdir(parents=True, exist_ok=True)
        try:
            (rd / f"{stem}.before.prom").write_text(prom_before)
            (rd / f"{stem}.after.prom").write_text(prom_after)
        except Exception as e:  # noqa: BLE001
            print(f"[replay] could not save prom snapshots: {e}", file=sys.stderr)

    # client-side metrics. client_e2e_ms is the ACTUAL end-to-end latency: per MEASURED
    # turn, request-send -> stream-complete, from raw client timestamps in replay_turn
    # (the untimed pre-arrange warmup turns are NOT in `e2es`, so this is steady-state).
    # EXACT percentiles over the real per-turn samples — unlike the server prom e2e
    # histogram, whose >60s buckets are 60s-wide and so estimate the tail coarsely. This
    # is the only true e2e; the prom e2e_ms in `server` is retained solely as a cross-check.
    median_itl = statistics.median(all_itls) if all_itls else None
    client = {
        "completed_turns": cached["ok"], "failed_turns": cached["fail"],
        "runtime_s": round(runtime, 2),
        "client_ttft_ms": {"p50": pct(ttfts, .5), "p99": pct(ttfts, .99)},
        "client_itl_ms": {"p50": pct(all_itls, .5), "p99": pct(all_itls, .99)},
        "client_e2e_ms": {
            "mean": round(statistics.mean(e2es), 3) if e2es else None,
            "p50": pct(e2es, .5), "p95": pct(e2es, .95), "p99": pct(e2es, .99),
        },
        "client_intvty_p50": round(1000 / median_itl, 2) if median_itl else None,
        "client_out_tokens": sum(outs),
    }
    # server prometheus truth (the headline numbers)
    server = M.summarize(prom_before, prom_after, runtime, args.n_gpu) if prom_before and prom_after else {}

    return {"concurrency": args.concurrency, "model": args.model,
            "prearrange_frac": args.prearrange_frac, "ramp_seconds": args.ramp_seconds,
            "server": server, "client": client}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--model", default="minimax")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--n-gpu", type=int, default=4, help="GPUs in the deployment (throughput is per-GPU)")
    ap.add_argument("--prearrange-lo", type=float, default=0.25,
                    help="lower bound of the random start-position band (Cam: 25-75%%)")
    ap.add_argument("--prearrange-frac", type=float, default=0.75,
                    help="upper bound of the start-position band; 0 disables pre-arrange")
    ap.add_argument("--ramp-seconds", type=float, default=60.0)
    ap.add_argument("--shuffle-sessions", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--extra-body", default=None)
    ap.add_argument("--result-dir", default="out")
    ap.add_argument("--result-filename", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))
    rd = Path(args.result_dir); rd.mkdir(parents=True, exist_ok=True)
    fn = args.result_filename or f"conc{args.concurrency}.json"
    (rd / fn).write_text(json.dumps(summary, indent=2))
    print(f"\n[replay] wrote {rd / fn}")


if __name__ == "__main__":
    main()
