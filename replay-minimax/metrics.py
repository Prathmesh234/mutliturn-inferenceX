#!/usr/bin/env python3
"""Prometheus-truth metric extraction for a colocated vLLM server.

Everything the benchmark reports is derived HERE from two `/metrics` snapshots
(one before, one after the measured window) — never from the client's own timing.
First principles:

  throughput  = generation_tokens Δ / runtime          (GPU efficiency)
  interactivity = 1000 / inter_token_latency_ms        (per-user speed)
  TTFT, E2E   = histogram quantiles over the window
  prefix-cache hit rate = prefix_cache_hits Δ / queries Δ   (KV reuse)
  GPU/CPU KV usage      = gauge (occupancy / offload pressure)

vLLM metric names drift slightly across versions, so each family lists aliases.
"""
from __future__ import annotations

import re

# family -> list of name aliases (first match wins). Same harness, both engines:
# vLLM uses the `vllm:` prefix, SGLang the `sglang:` prefix.
TTFT = ["vllm:time_to_first_token_seconds", "sglang:time_to_first_token_seconds"]
# per-token-GAP inter-token latency: one histogram observation per generated token
# gap (count ~= output_tokens). Reflects the decode step time distribution.
ITL = ["vllm:inter_token_latency_seconds", "sglang:inter_token_latency_seconds"]
# per-REQUEST time-per-output-token: one observation per request (count == #requests);
# this is the canonical TPOT. vLLM exposes `request_time_per_output_token_seconds`
# (the older bare `time_per_output_token_seconds` name does NOT exist on this build, so
# it was a dead alias). SGLang has no per-request TPOT histogram -> lat_stats returns
# None and callers fall back to ITL.
TPOT = ["vllm:request_time_per_output_token_seconds",
        "sglang:request_time_per_output_token_seconds",
        "sglang:time_per_output_token_seconds"]
E2E = ["vllm:e2e_request_latency_seconds", "vllm:request_latency_seconds",
       "sglang:e2e_request_latency_seconds"]
GEN_TOK = ["vllm:generation_tokens_total", "sglang:generation_tokens_total"]
PROMPT_TOK = ["vllm:prompt_tokens_total", "sglang:prompt_tokens_total"]
# vLLM v0.21.0 dropped the `gpu_` prefix on these counters (older builds had it);
# list both so the harness works across versions. Counter-delta is the truth.
CACHE_HITS = ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"]
CACHE_QUERIES = ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"]
# gauge fallback (instantaneous): vLLM versions + SGLang (no cumulative counter exposed)
CACHE_HIT_RATE = ["vllm:gpu_prefix_cache_hit_rate", "vllm:prefix_cache_hit_rate",
                  "sglang:cache_hit_rate"]
GPU_KV = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc", "sglang:token_usage"]
CPU_KV = ["vllm:cpu_cache_usage_perc"]
RUNNING = ["vllm:num_requests_running", "sglang:num_running_reqs"]
WAITING = ["vllm:num_requests_waiting", "sglang:num_queue_reqs"]


def parse(text: str):
    """Return (scalars, hists). scalars[name]=summed value (across label sets);
    hists[name] = {'sum':x,'count':n,'buckets':[(le,cum)]}."""
    scalars: dict[str, float] = {}
    hsum: dict[str, float] = {}
    hcount: dict[str, float] = {}
    hbkt: dict[str, dict[float, float]] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = re.match(r"(\S+?)(\{[^}]*\})?\s+([0-9eE+.\-]+)\s*$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name.endswith("_bucket"):
            base = name[:-7]
            le = re.search(r'le="([^"]+)"', labels)
            if le:
                lev = float("inf") if le.group(1) in ("+Inf", "Inf") else float(le.group(1))
                hbkt.setdefault(base, {})
                hbkt[base][lev] = hbkt[base].get(lev, 0.0) + val
        elif name.endswith("_sum"):
            hsum[name[:-4]] = hsum.get(name[:-4], 0.0) + val
        elif name.endswith("_count"):
            hcount[name[:-6]] = hcount.get(name[:-6], 0.0) + val
        else:
            scalars[name] = scalars.get(name, 0.0) + val
    hists = {}
    for base in set(hsum) | set(hcount) | set(hbkt):
        bkts = sorted(hbkt.get(base, {}).items())
        hists[base] = {"sum": hsum.get(base, 0.0), "count": hcount.get(base, 0.0), "buckets": bkts}
    return scalars, hists


def _first(d: dict, aliases):
    for a in aliases:
        if a in d:
            return d[a]
    return None


def quantile(buckets, total, q):
    """Prometheus histogram quantile from cumulative [(le, cum)] buckets."""
    if total <= 0 or not buckets:
        return 0.0
    rank = q * total
    prev_le, prev_cum = 0.0, 0.0
    for le, cum in buckets:
        if cum >= rank:
            if le == float("inf"):
                return prev_le
            if cum == prev_cum:
                return le
            return prev_le + (rank - prev_cum) / (cum - prev_cum) * (le - prev_le)
        if le != float("inf"):
            prev_le = le
        prev_cum = cum
    return prev_le


def _hist_delta(b_hists, a_hists, aliases):
    """Difference an histogram family across before/after snapshots -> (sum,count,buckets)."""
    for name in aliases:
        if name in a_hists:
            a, b = a_hists[name], b_hists.get(name, {"sum": 0, "count": 0, "buckets": []})
            bmap = dict(b["buckets"])
            les = sorted({le for le, _ in a["buckets"]})
            buckets = [(le, dict(a["buckets"]).get(le, 0) - bmap.get(le, 0)) for le in les]
            return a["sum"] - b["sum"], a["count"] - b["count"], buckets
    return 0.0, 0.0, []


def lat_stats(b_hists, a_hists, aliases):
    """Return (mean_ms, p50_ms, p99_ms) for a seconds-histogram over the window."""
    s, c, bkts = _hist_delta(b_hists, a_hists, aliases)
    if c <= 0:
        return None
    return (round(s / c * 1000, 3),
            round(quantile(bkts, c, 0.50) * 1000, 3),
            round(quantile(bkts, c, 0.99) * 1000, 3))


def summarize(before_text: str, after_text: str, runtime_s: float, n_gpu: int) -> dict:
    """The prom-truth metric bundle for one measured window."""
    bs, bh = parse(before_text)
    as_, ah = parse(after_text)

    def cdelta(aliases):
        bv, av = _first(bs, aliases), _first(as_, aliases)
        return (av - bv) if (av is not None and bv is not None) else None

    gen = cdelta(GEN_TOK) or 0.0
    prompt = cdelta(PROMPT_TOK) or 0.0
    out_tps = gen / runtime_s if runtime_s else 0.0
    in_tps = prompt / runtime_s if runtime_s else 0.0

    ttft = lat_stats(bh, ah, TTFT)
    itl = lat_stats(bh, ah, ITL)
    tpot = lat_stats(bh, ah, TPOT)        # per-request TPOT; None on SGLang
    e2e = lat_stats(bh, ah, E2E)
    # interactivity (tok/s/user) is 1/TPOT; prefer the per-request TPOT metric, fall
    # back to per-gap ITL where the engine doesn't expose TPOT (SGLang).
    tp = tpot or itl

    def intvty(v_ms):
        return round(1000.0 / v_ms, 2) if v_ms else None

    # prefix cache hit rate: prefer hits/queries delta; else the gauge (use 'after')
    hits, queries = cdelta(CACHE_HITS), cdelta(CACHE_QUERIES)
    if hits is not None and queries and queries > 0:
        cache_hit = round(hits / queries, 4)
    else:
        g = _first(as_, CACHE_HIT_RATE)
        cache_hit = round(g, 4) if g is not None else None

    return {
        "runtime_s": round(runtime_s, 2),
        "n_gpu": n_gpu,
        # throughput
        "output_tput_per_gpu": round(out_tps / n_gpu, 2),
        "input_tput_per_gpu": round(in_tps / n_gpu, 2),
        "output_tput_total": round(out_tps, 2),
        # interactivity (per user) = 1/TPOT (per-request); ITL fallback on SGLang
        "intvty_p50": intvty(tp[1]) if tp else None,
        "intvty_p99": intvty(tp[2]) if tp else None,
        # latency (ms) mean / p50 / p99
        "ttft_ms": {"mean": ttft[0], "p50": ttft[1], "p99": ttft[2]} if ttft else None,
        "itl_ms": {"mean": itl[0], "p50": itl[1], "p99": itl[2]} if itl else None,
        "tpot_ms": {"mean": tpot[0], "p50": tpot[1], "p99": tpot[2]} if tpot else None,
        "e2e_ms": {"mean": e2e[0], "p50": e2e[1], "p99": e2e[2]} if e2e else None,
        # cache / capacity (the stage-2 metrics)
        "gpu_prefix_cache_hit_rate": cache_hit,
        "gpu_kv_usage_perc": _first(as_, GPU_KV),
        "cpu_kv_usage_perc": _first(as_, CPU_KV),
        "num_running": _first(as_, RUNNING),
        "num_waiting": _first(as_, WAITING),
    }
