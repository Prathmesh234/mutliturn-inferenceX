#!/usr/bin/env python3
"""Build the InferenceX-app local-dump for the MiniMax-M2.5 2x2 serving study.

Single model (MiniMax-M2.5), one node (4x GB300), real multi-turn Claude Code
agentic replay. The 2x2 isolates **expert parallelism** across two engines:

    framework key   run dir                    engine   MoE parallelism
    -------------   -----------------------    ------   ----------------
    vllm-ep         agg_vllm_tp4               vLLM     EP4 (--enable-expert-parallel)
    vllm-tp         agg_vllm_tp4_noep          vLLM     TP-only (control)
    sglang-tp       agg_sglang_tp4             SGLang   TP-only (control)
    sglang-ep       agg_sglang_tp4_ep4         SGLang   EP4 (--ep-size 4)

Each is a DISTINCT framework key so the app renders 4 separate rooflines/colors
(the app separates series by gpu_framework; EP vs TP is not otherwise a series
axis). All four share: MiniMax-M2.5, 4x GB300, TP4, fp4 (NVFP4 weights + fp8 KV),
aggregated/colocated, prefix-caching on, context 40960.

SOURCE = SERVER PROMETHEUS TRUTH. sweep.py parsed each engine's /metrics across
the steady-state window and wrote per-conc numbers into conc<N>.json -> `server`
(mean/p50/p99 for ttft, itl(=tpot), e2e; per-GPU throughput; interactivity). We
read that block verbatim. No client timing, no fabrication. Whatever concs exist
on disk are emitted (sglang-ep may still be sweeping -> partial).

Tail = p99 everywhere (the app does not consume p90; p90_* mirrors p99_*).
Workload ISL/OSL = MEAN over all 1769 replayed turns (37109 / 1490).

Usage:  python make_inferencex_dump_minimax.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path("/mnt/home/ppbhatt500/gpumode-triton/results_minimax")
CONCS = (4, 8, 16, 32, 64, 128)
DATE = "2026-06-09"

MODEL = "minimaxm2.5"   # DB key -> display "MiniMax-M2.5"
PRECISION = "fp4"       # NVFP4 weights (+ fp8 KV)
HARDWARE = "gb300"
TP = 4
N_GPU = 4               # colocated: prefill+decode share the 4 GPUs
ISL = 37109             # mean ISL over the 1769 replayed turns
OSL = 1490              # mean OSL

#   framework key (=> distinct series), run dir, EP size (TP-only => ep 1)
RUNS = [
    dict(fw="vllm-ep",   dir="agg_vllm_tp4",        ep=4),
    dict(fw="vllm-tp",   dir="agg_vllm_tp4_noep",   ep=1),
    dict(fw="sglang-tp", dir="agg_sglang_tp4",      ep=1),
    dict(fw="sglang-ep", dir="agg_sglang_tp4_ep4",  ep=4),
]


def metrics_from_server(srv: dict) -> dict:
    """conc<N>.json `server` block (prom truth) -> app metrics schema.

    App stores latencies in SECONDS (conc.json is ms -> /1000). Interactivity
    (tok/s/user) = 1000 / itl_ms. itl == tpot. Tail = p99; p90_* mirrors p99_*.
    """
    ttft, itl, e2e = srv["ttft_ms"], srv["itl_ms"], srv["e2e_ms"]

    def s(ms):
        return round(ms / 1000.0, 6)

    def intvty(itl_ms):
        return round(1000.0 / itl_ms, 4) if itl_ms else 0.0

    out_pg = round(srv["output_tput_per_gpu"], 4)
    in_pg = round(srv["input_tput_per_gpu"], 4)
    return {
        "tput_per_gpu": out_pg, "output_tput_per_gpu": out_pg, "input_tput_per_gpu": in_pg,
        "mean_ttft": s(ttft["mean"]), "median_ttft": s(ttft["p50"]),
        "std_ttft": 0.0, "p90_ttft": s(ttft["p99"]), "p99_ttft": s(ttft["p99"]),
        "mean_tpot": s(itl["mean"]), "median_tpot": s(itl["p50"]),
        "std_tpot": 0.0, "p90_tpot": s(itl["p99"]), "p99_tpot": s(itl["p99"]),
        "mean_itl": s(itl["mean"]), "median_itl": s(itl["p50"]),
        "std_itl": 0.0, "p90_itl": s(itl["p99"]), "p99_itl": s(itl["p99"]),
        "mean_intvty": intvty(itl["mean"]), "median_intvty": intvty(itl["p50"]),
        "std_intvty": 0.0, "p90_intvty": intvty(itl["p99"]), "p99_intvty": intvty(itl["p99"]),
        "mean_e2el": s(e2e["mean"]), "median_e2el": s(e2e["p50"]),
        "std_e2el": 0.0, "p90_e2el": s(e2e["p99"]), "p99_e2el": s(e2e["p99"]),
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "InferenceX-app" / "local-dump"
    out.mkdir(parents=True, exist_ok=True)

    configs, benchmarks, availability = [], [], []
    summary = []
    bid = 1
    for cid, run in enumerate(RUNS, start=1):
        ep = run["ep"]
        configs.append({
            "id": cid, "hardware": HARDWARE, "framework": run["fw"], "model": MODEL,
            "precision": PRECISION, "spec_method": "none", "disagg": False, "is_multinode": False,
            "prefill_tp": TP, "prefill_ep": ep, "prefill_dp_attention": False, "prefill_num_workers": 1,
            "decode_tp": TP, "decode_ep": ep, "decode_dp_attention": False, "decode_num_workers": 1,
            "num_prefill_gpu": N_GPU, "num_decode_gpu": N_GPU,
        })
        availability.append({
            "model": MODEL, "isl": ISL, "osl": OSL, "precision": PRECISION,
            "hardware": HARDWARE, "framework": run["fw"], "spec_method": "none",
            "disagg": False, "date": DATE,
        })
        got = []
        for conc in CONCS:
            f = RESULTS / run["dir"] / f"conc{conc}.json"
            if not f.exists():
                continue
            srv = json.loads(f.read_text())["server"]
            benchmarks.append({
                "id": bid, "workflow_run_id": 1, "config_id": cid, "benchmark_type": "agentic-replay",
                "date": DATE, "isl": ISL, "osl": OSL, "conc": conc, "image": None,
                "metrics": metrics_from_server(srv), "workers": None, "error": None, "server_log_id": None,
            })
            got.append((conc, round(srv["output_tput_per_gpu"], 1)))
            bid += 1
        summary.append((run["fw"], run["dir"], ep, got))

    workflow_runs = [{
        "id": 1, "github_run_id": 1, "run_attempt": 1,
        "name": "MiniMax-M2.5 agentic replay 2x2 — vLLM/SGLang × EP/TP-only (SERVER PROM TRUTH, 4xGB300 colocated)",
        "status": "completed", "conclusion": "success", "head_sha": "local",
        "head_branch": "master", "html_url": None,
        "created_at": f"{DATE}T00:00:00.000Z", "run_started_at": f"{DATE}T00:00:00.000Z", "date": DATE,
    }]

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

    print("\n=== MiniMax-M2.5 2x2 (4xGB300, colocated, fp4+fp8KV, ISL/OSL 37109/1490) — tok/s/GPU ===")
    print(f"  {'framework':10} {'ep':>2}  " + " ".join(f"{c:>7}" for c in CONCS))
    for fw, d, ep, got in summary:
        m = {c: v for c, v in got}
        row = " ".join(f"{m[c]:>7}" if c in m else f"{'·':>7}" for c in CONCS)
        tag = "" if len(got) == len(CONCS) else f"  (partial {len(got)}/6)"
        print(f"  {fw:10} {ep:>2}  {row}{tag}")
    print(f"\ndump dir: {out}  | configs={len(configs)} benchmarks={len(benchmarks)} availability={len(availability)}")


if __name__ == "__main__":
    main()
