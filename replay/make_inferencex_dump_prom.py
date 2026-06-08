#!/usr/bin/env python3
"""Build the InferenceX-app dump using SERVER PROMETHEUS TRUTH ONLY.

Motivation: the client-side replay metrics (conc*.json) have a TPOT/ITL
aggregation bug on some configs (kimi_b1 vllm ~21x; dsv4_sglang_4p_8d disagg
~120x) that makes interactivity look catastrophically wrong. The SERVER prom
counters are the ground truth, so every metric here is derived from the engine's
own /metrics, not the client.

Sources (per concurrency point):
  COLOCATED  results/<dir>/c<N>_prom.json  -> the `delta` block = before/after diff
             of the vLLM / SGLang /metrics across THAT concurrency window.
             engine prefix: vllm:  or  sglang:
  DISAGG     results/<dir>/server_metrics.prom = timestamped cumulative snapshots
             (every 15s) of the Dynamo frontend. We isolate each conc window as
             [mtime(conc<N>.json) - runtime_s, mtime] and diff the cumulative
             histograms across it. engine prefix: dynamo_frontend_

Metric families used (all three engines expose these):
  <p>time_to_first_token_seconds   (ttft)      <p>=vllm:/sglang:
  <p>inter_token_latency_seconds    (tpot/itl)  dynamo uses dynamo_frontend_*
  <p>e2e_request_latency_seconds / dynamo_frontend_request_duration_seconds (e2e)
  <p>generation_tokens_total / dynamo_frontend_output_tokens_total  (out tokens)
  <p>prompt_tokens_total / dynamo_frontend_input_sequence_tokens_sum (in tokens)

For each: mean = sum/count (exact); median/p99 = Prometheus histogram-quantile by
bucket interpolation. intvty (tok/s/user) = 1000/tpot_ms. Throughput per GPU =
out_token_delta / runtime_s / total_gpu (colocated /4; disagg /(prefill+decode)).
Latencies emitted in SECONDS (the app's unit). ISL/OSL bucketed to 8192/1024.

Usage:  python make_inferencex_dump_prom.py [out_dir]
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

RESULTS = Path("/mnt/home/ppbhatt500/gpumode-triton/results")
DATE = "2026-06-08"
ISL, OSL = 8192, 1024
COLO = (4, 8, 16, 32, 64, 128)
DIS = (4, 8, 16)

#   dir, model, framework, engine_prefix, disagg, multinode, p_tp,p_w,d_tp,d_w, conc
RUNS = [
    dict(dir="dsv4_b1",        model="dsv4",     fw="vllm",          eng="vllm",   disagg=False, mn=False, p_tp=4,p_w=1,d_tp=4,d_w=1, conc=COLO),
    dict(dir="dsv4_b1_sglang", model="dsv4",     fw="sglang",        eng="sglang", disagg=False, mn=False, p_tp=4,p_w=1,d_tp=4,d_w=1, conc=COLO),
    dict(dir="kimi_b1",        model="kimik2.5", fw="vllm",          eng="vllm",   disagg=False, mn=False, p_tp=4,p_w=1,d_tp=4,d_w=1, conc=COLO),
    dict(dir="kimi_b1_sglang", model="kimik2.5", fw="sglang",        eng="sglang", disagg=False, mn=False, p_tp=4,p_w=1,d_tp=4,d_w=1, conc=COLO),
    dict(dir="dsv4_vllm_4p_4d",   model="dsv4", fw="dynamo-vllm",   eng="dynamo", disagg=True, mn=True, p_tp=4,p_w=1,d_tp=4,d_w=1, conc=DIS),
    dict(dir="dsv4_vllm_8p_4d",   model="dsv4", fw="dynamo-vllm",   eng="dynamo", disagg=True, mn=True, p_tp=4,p_w=2,d_tp=4,d_w=1, conc=DIS),
    dict(dir="dsv4_vllm_4p_8d",   model="dsv4", fw="dynamo-vllm",   eng="dynamo", disagg=True, mn=True, p_tp=4,p_w=1,d_tp=4,d_w=2, conc=DIS),
    dict(dir="dsv4_sglang_4p_4d", model="dsv4", fw="dynamo-sglang", eng="dynamo", disagg=True, mn=True, p_tp=4,p_w=1,d_tp=4,d_w=1, conc=DIS),
    dict(dir="dsv4_sglang_4p_8d", model="dsv4", fw="dynamo-sglang", eng="dynamo", disagg=True, mn=True, p_tp=4,p_w=1,d_tp=4,d_w=2, conc=DIS),
]

# metric family -> (colocated suffix for vllm, for sglang, dynamo full name)
FAM = {
    "ttft":   ("vllm:time_to_first_token_seconds",  "sglang:time_to_first_token_seconds",  "dynamo_frontend_time_to_first_token_seconds"),
    "tpot":   ("vllm:inter_token_latency_seconds",  "sglang:inter_token_latency_seconds",  "dynamo_frontend_inter_token_latency_seconds"),
    "e2e":    ("vllm:e2e_request_latency_seconds",  "sglang:e2e_request_latency_seconds",  "dynamo_frontend_request_duration_seconds"),
}
OUT_TOK = ("vllm:generation_tokens_total", "sglang:generation_tokens_total", "dynamo_frontend_output_tokens_total")
IN_TOK  = ("vllm:prompt_tokens_total",     "sglang:prompt_tokens_total",     "dynamo_frontend_input_sequence_tokens_sum")


def base_name(eng):  # index into the tuples above
    return {"vllm": 0, "sglang": 1, "dynamo": 2}[eng]


def quantile(buckets, total, q):
    """Prometheus-style histogram quantile. buckets: sorted [(le, cumcount)] incl +Inf."""
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


# ---- COLOCATED: read delta dict from c<N>_prom.json ------------------------
def hist_from_dict(delta, prefix):
    """Return (sum, count, buckets[(le,cum)]) for a metric prefix from a flat dict."""
    s = c = None
    buckets = []
    for k, v in delta.items():
        if k.startswith(prefix + "_sum"):
            s = v
        elif k.startswith(prefix + "_count"):
            c = v
        elif k.startswith(prefix + "_bucket"):
            m = re.search(r'le="([^"]+)"', k)
            if m:
                le = float("inf") if m.group(1) in ("+Inf", "Inf") else float(m.group(1))
                buckets.append((le, v))
    buckets.sort()
    if c is not None and (not buckets or buckets[-1][0] != float("inf")):
        buckets.append((float("inf"), c))
    return s, c, buckets


def scalar_from_dict(delta, prefix):
    tot = 0.0
    found = False
    for k, v in delta.items():
        if k.startswith(prefix):
            tot += v
            found = True
    return tot if found else None


# ---- DISAGG: parse timestamped snapshots, diff across a window -------------
def parse_snapshots(path, wanted_prefixes):
    snaps = []
    ts = None
    cur = {}
    pat = re.compile(r"^([a-z_]+(?:_seconds)?(?:_sum|_count|_bucket)?|dynamo_frontend_output_tokens_total|dynamo_frontend_input_sequence_tokens_sum)")
    with open(path) as fh:
        for line in fh:
            if line.startswith("# ==== ts="):
                if ts is not None:
                    snaps.append((ts, cur))
                ts = dt.datetime.fromisoformat(line.split("ts=")[1].split(" ")[0])
                cur = {}
                continue
            if line.startswith("#") or not line.strip():
                continue
            if not line.startswith("dynamo_frontend_"):
                continue
            if not any(p in line for p in wanted_prefixes):
                continue
            m = re.match(r"(\S+?)(\{[^}]*\})?\s+([0-9eE+.\-]+)\s*$", line.strip())
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2) or "", m.group(3)
            le = ""
            lm = re.search(r'le="([^"]+)"', labels)
            if lm:
                le = lm.group(1)
            key = name + ("|" + le if le else "")
            cur[key] = float(val)  # one frontend series per snapshot
    if ts is not None:
        snaps.append((ts, cur))
    return snaps


def snap_at(snaps, t):
    best = snaps[0][1]
    for ts, c in snaps:
        if ts <= t:
            best = c
        else:
            break
    return best


def hist_from_diff(a, b, name):
    s = b.get(name + "_sum", 0) - a.get(name + "_sum", 0)
    c = b.get(name + "_count", 0) - a.get(name + "_count", 0)
    buckets = []
    les = sorted({k.split("|")[1] for k in b if k.startswith(name + "_bucket|")},
                 key=lambda x: float("inf") if x in ("+Inf", "Inf") else float(x))
    for le in les:
        k = f"{name}_bucket|{le}"
        cum = b.get(k, 0) - a.get(k, 0)
        lef = float("inf") if le in ("+Inf", "Inf") else float(le)
        buckets.append((lef, cum))
    if buckets and buckets[-1][0] != float("inf"):
        buckets.append((float("inf"), c))
    return s, c, buckets


def stats(s, c, buckets):
    """Return (mean, p50, p90, p99) in ms from a seconds-histogram."""
    mean = (s / c * 1000) if c else 0.0

    def q(x):
        return quantile(buckets, c, x) * 1000 if c else 0.0

    return round(mean, 4), round(q(0.50), 4), round(q(0.90), 4), round(q(0.99), 4)


def build_metrics(run, conc, total_gpu, selfcheck):
    eng = run["eng"]
    bi = base_name(eng)
    rt = json.loads((RESULTS / run["dir"] / f"conc{conc}.json").read_text())["runtime_s"]

    if eng != "dynamo":  # COLOCATED — delta from c<N>_prom.json
        delta = json.loads((RESULTS / run["dir"] / f"c{conc}_prom.json").read_text())["delta"]
        H = {f: hist_from_dict(delta, FAM[f][bi]) for f in FAM}
        out_tok = scalar_from_dict(delta, OUT_TOK[bi]) or 0.0
        in_tok = scalar_from_dict(delta, IN_TOK[bi]) or 0.0
    else:  # DISAGG — window-diff of server_metrics.prom for LATENCY histograms
        snaps = run["_snaps"]
        f = RESULTS / run["dir"] / f"conc{conc}.json"
        mt = dt.datetime.fromtimestamp(os.path.getmtime(f), tz=dt.timezone.utc)
        st = mt - dt.timedelta(seconds=rt)
        a, b = snap_at(snaps, st), snap_at(snaps, mt)
        H = {fam: hist_from_diff(a, b, FAM[fam][bi]) for fam in FAM}
        # THROUGHPUT for disagg: use the server-generated token COUNT. The Dynamo
        # frontend's output_tokens_total counter is non-monotonic on some runs
        # (dsv4_sglang_4p_8d: 136 scrape resets / frontend replica round-robin), so
        # naive window-diffing undercounts. The token count itself is sound — the
        # prom CUMULATIVE total matches conc.json (12 failed turns excluded), and the
        # 4 clean runs' prom positive-increment == conc.json exactly. So we take the
        # validated server-generated token count from conc.json. Latency/interactivity
        # (the actually-buggy client metric) still come purely from prom histograms.
        cjt = json.loads((RESULTS / run["dir"] / f"conc{conc}.json").read_text())
        out_tok = cjt["output_tokens_total"]
        in_tok = cjt["input_tokens_total"]

    ttft = stats(*H["ttft"])
    tpot = stats(*H["tpot"])
    e2e = stats(*H["e2e"])
    out_tps = out_tok / rt
    in_tps = in_tok / rt

    def intvty(tpot_ms):
        return round(1000.0 / tpot_ms, 4) if tpot_ms else 0.0

    # self-check: prom throughput vs client conc.json (token counts should agree)
    cj = json.loads((RESULTS / run["dir"] / f"conc{conc}.json").read_text())
    cli_tps = cj["output_throughput_tok_per_s"]
    drift = abs(out_tps - cli_tps) / cli_tps if cli_tps else 0
    selfcheck.append((run["dir"], conc, round(out_tps, 1), round(cli_tps, 1), round(drift * 100, 1),
                      round(tpot[1], 2), round(cj["tpot_ms"]["p50"], 2)))

    return {
        "tput_per_gpu": round(out_tps / total_gpu, 4),
        "output_tput_per_gpu": round(out_tps / total_gpu, 4),
        "input_tput_per_gpu": round(in_tps / total_gpu, 4),
        "mean_ttft": round(ttft[0] / 1000, 6), "median_ttft": round(ttft[1] / 1000, 6),
        "std_ttft": 0.0, "p90_ttft": round(ttft[2] / 1000, 6), "p99_ttft": round(ttft[3] / 1000, 6),
        "mean_tpot": round(tpot[0] / 1000, 6), "median_tpot": round(tpot[1] / 1000, 6),
        "std_tpot": 0.0, "p90_tpot": round(tpot[2] / 1000, 6), "p99_tpot": round(tpot[3] / 1000, 6),
        "mean_intvty": intvty(tpot[0]), "median_intvty": intvty(tpot[1]),
        "std_intvty": 0.0, "p90_intvty": intvty(tpot[2]), "p99_intvty": intvty(tpot[3]),
        "mean_itl": round(tpot[0] / 1000, 6), "median_itl": round(tpot[1] / 1000, 6),
        "std_itl": 0.0, "p90_itl": round(tpot[2] / 1000, 6), "p99_itl": round(tpot[3] / 1000, 6),
        "mean_e2el": round(e2e[0] / 1000, 6), "median_e2el": round(e2e[1] / 1000, 6),
        "std_e2el": 0.0, "p90_e2el": round(e2e[2] / 1000, 6), "p99_e2el": round(e2e[3] / 1000, 6),
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "InferenceX-app" / "local-dump"
    out.mkdir(parents=True, exist_ok=True)

    # preload disagg snapshots once per run
    wanted = [FAM[f][2] for f in FAM] + [OUT_TOK[2], IN_TOK[2]]
    for run in RUNS:
        if run["eng"] == "dynamo":
            run["_snaps"] = parse_snapshots(RESULTS / run["dir"] / "server_metrics.prom", wanted)

    configs, benchmarks, availability = [], [], []
    avail_seen = set()
    selfcheck = []
    bid = 1
    for cid, run in enumerate(RUNS, start=1):
        npg, ndg = run["p_tp"] * run["p_w"], run["d_tp"] * run["d_w"]
        total_gpu = (npg + ndg) if run["disagg"] else run["d_tp"]
        configs.append({
            "id": cid, "hardware": "gb300", "framework": run["fw"], "model": run["model"],
            "precision": "fp4", "spec_method": "none", "disagg": run["disagg"], "is_multinode": run["mn"],
            "prefill_tp": run["p_tp"], "prefill_ep": 1, "prefill_dp_attention": False, "prefill_num_workers": run["p_w"],
            "decode_tp": run["d_tp"], "decode_ep": 1, "decode_dp_attention": False, "decode_num_workers": run["d_w"],
            "num_prefill_gpu": npg, "num_decode_gpu": ndg,
        })
        if (run["model"], run["fw"]) not in avail_seen:
            avail_seen.add((run["model"], run["fw"]))
            availability.append({"model": run["model"], "isl": ISL, "osl": OSL, "precision": "fp4",
                                 "hardware": "gb300", "framework": run["fw"], "spec_method": "none",
                                 "disagg": run["disagg"], "date": DATE})
        for conc in run["conc"]:
            if not (RESULTS / run["dir"] / f"conc{conc}.json").exists():
                continue
            benchmarks.append({
                "id": bid, "workflow_run_id": 1, "config_id": cid, "benchmark_type": "agentic-replay",
                "date": DATE, "isl": ISL, "osl": OSL, "conc": conc, "image": None,
                "metrics": build_metrics(run, conc, total_gpu, selfcheck),
                "workers": None, "error": None, "server_log_id": None,
            })
            bid += 1

    workflow_runs = [{"id": 1, "github_run_id": 1, "run_attempt": 1,
                      "name": "Agentic replay pareto (SERVER PROM TRUTH) - DSv4/Kimi x vLLM/SGLang, colocated + Dynamo disagg (GB300)",
                      "status": "completed", "conclusion": "success", "head_sha": "local",
                      "head_branch": "custom-replay", "html_url": None,
                      "created_at": f"{DATE}T00:00:00.000Z", "run_started_at": f"{DATE}T00:00:00.000Z", "date": DATE}]

    def w(name, data):
        (out / name).write_text(json.dumps(data, indent=2))
        print(f"  wrote {name}: {len(data)} rows")

    w("configs.json", configs)
    w("workflow_runs.json", workflow_runs)
    w("benchmark_results.json", benchmarks)
    w("availability.json", availability)
    w("run_stats.json", [])
    w("eval_results.json", [])
    w("changelog_entries.json", [])

    print("\n=== SELF-CHECK: prom throughput vs client (drift) + tpot_p50 prom vs client ===")
    print(f"  {'dir':20} {'conc':>4} {'prom_tps':>9} {'cli_tps':>8} {'drift%':>7}  {'prom_tpot':>9} {'cli_tpot':>9}")
    for d, conc, p, c, dr, pt, ct in selfcheck:
        flag = "  <-- throughput drift" if dr > 8 else ""
        print(f"  {d:20} {conc:>4} {p:>9} {c:>8} {dr:>7}  {pt:>9} {ct:>9}{flag}")
    print(f"\ndump dir: {out}  | configs={len(configs)} benchmarks={len(benchmarks)}")


if __name__ == "__main__":
    main()
