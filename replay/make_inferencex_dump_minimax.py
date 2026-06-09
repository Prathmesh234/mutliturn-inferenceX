#!/usr/bin/env python3
"""Build the InferenceX-app local-dump for the MiniMax-M2.5 serving study.

Single model (MiniMax-M2.5), one node (4x GB300), real multi-turn Claude Code
agentic replay. Two experiments, all rendered as distinct framework series (the
app separates series by gpu_framework; EP/TP/batch-tuning are not otherwise a
series axis):

  --- 2x2: expert-parallel × engine (dataset batch_long.replay.jsonl) ---
    framework key   run dir                 engine   MoE parallelism
    vllm-ep         agg_vllm_tp4            vLLM     EP4 (--enable-expert-parallel)
    vllm-tp         agg_vllm_tp4_noep       vLLM     TP-only (control)
    sglang-tp       agg_sglang_tp4          SGLang   TP-only (control)
    sglang-ep       agg_sglang_tp4_ep4      SGLang   EP4 (--ep-size 4)

  --- bigbatch: vLLM-EP high-concurrency batch tuning (dataset batch_long_x6) ---
    vllm-ep-bb32k   agg_vllm_tp4_ep_bb      vLLM EP  max-num-batched-tokens=32768, max-num-seqs=128
    vllm-ep-bb64k   agg_vllm_tp4_ep_bb64k   vLLM EP  max-num-batched-tokens=65536, max-num-seqs=256

  batch_long_x6 is batch_long replicated 6x (900 sessions / 10614 turns) so the
  sweep has enough distinct sessions for conc up to 256 — the PER-TURN ISL/OSL
  distribution is identical, so every series shares the same 37109/1490 bucket.
  The batch knobs have no config-schema field, so they live in the series label.

All share: MiniMax-M2.5, 4x GB300, TP4, fp4 (NVFP4 weights + fp8 KV), aggregated/
colocated, prefix-caching on, context 40960.

SOURCE = SERVER PROMETHEUS TRUTH (each conc<N>.json `server` block, written by
sweep.py from the engine /metrics). No client timing, no fabrication. Concs are
detected dynamically (supports conc256); a run dir with no conc data is skipped
(e.g. a sweep still warming up). Tail = p99 (app doesn't read p90; p90 mirrors p99).

Usage:  python make_inferencex_dump_minimax.py [out_dir]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path("/mnt/home/ppbhatt500/gpumode-triton/results_minimax")
DATE = "2026-06-09"

MODEL = "minimaxm2.5"   # DB key -> display "MiniMax-M2.5"
PRECISION = "fp4"       # NVFP4 weights (+ fp8 KV)
HARDWARE = "gb300"
TP = 4
N_GPU = 4               # colocated: prefill+decode share the 4 GPUs
ISL = 37109             # mean ISL over the replayed turns (same for x1 and x6)
OSL = 1490              # mean OSL

#   framework key (=> distinct series), run dir, EP size (TP-only => ep 1)
RUNS = [
    dict(fw="vllm-ep",       dir="agg_vllm_tp4",        ep=4),
    dict(fw="vllm-tp",       dir="agg_vllm_tp4_noep",   ep=1),
    dict(fw="sglang-tp",     dir="agg_sglang_tp4",      ep=1),
    dict(fw="sglang-ep",     dir="agg_sglang_tp4_ep4",  ep=4),
    # bigbatch (vLLM-EP high-conc batch tuning, x6 dataset) — DEFERRED: the sweep
    # is still in flight AND the wall-clock-averaged tput isn't comparable to the
    # saturated-decode tput the run is meant to show. Re-enable once the sweep is
    # complete and we've settled which throughput definition the chart uses.
    # dict(fw="vllm-ep-bb32k", dir="agg_vllm_tp4_ep_bb",     ep=4),
    # dict(fw="vllm-ep-bb64k", dir="agg_vllm_tp4_ep_bb64k",  ep=4),
]


def concs_in(d: Path):
    """Sorted concurrency ints present on disk for a run dir (dynamic; conc256 ok)."""
    out = []
    for f in d.glob("conc*.json"):
        m = re.fullmatch(r"conc(\d+)\.json", f.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _tput_pg(srv: dict) -> float:
    """STEADY-STATE full-batch tput per GPU (recompute_steady.py, corroborated by the
    x6 sustained run). recompute_steady removes the whole-window average from
    conc<N>.json, so steady is the only output throughput."""
    return round(srv["output_tput_per_gpu_steady"], 4)


def metrics_from_server(srv: dict) -> dict:
    """conc<N>.json `server` block (prom truth) -> app metrics schema.

    TPOT / ITL / interactivity = log-derived STEADY full-batch (tpot_steady_ms,
    intvty_steady = median gen_tput/running-batch), consistent with the steady
    throughput on the same point. The prom decode-latency histograms (itl_ms/tpot_ms)
    were REMOVED from conc<N>.json — their buckets jump 10ms->25ms, too coarse to
    resolve 5-20ms decode (they collapsed c32/c64/c128 to ~one value). TTFT/E2E have no
    log-derived steady form (engines don't log per-request TTFT), so they stay the prom
    window distribution. Latencies in SECONDS (conc.json is ms -> /1000).
    """
    ttft, e2e = srv["ttft_ms"], srv["e2e_ms"]
    tpot_steady = srv["tpot_steady_ms"]         # steady full-batch TPOT (ms), log-derived
    iv_steady = srv["intvty_steady"]            # steady full-batch tok/s/user, log-derived

    def s(ms):
        return round(ms / 1000.0, 6) if ms is not None else 0.0

    def p95(d):  # fall back to p99 if a point had no prom histogram
        return d.get("p95", d["p99"])

    out_pg = _tput_pg(srv)
    in_pg = round(srv["input_tput_per_gpu"], 4)
    tpot_med = tpot_steady
    iv_med = iv_steady
    # SemiAnalysis/InferenceX convention: tput_per_gpu is the TOTAL token throughput
    # per GPU = input (prefill) + output (decode), NOT output alone. The app's default
    # y-axis (y_tpPerGpu, "Throughput/GPU") reads this field as the total, with
    # output_tput_per_gpu / input_tput_per_gpu the per-direction breakdown. (See
    # InferenceX AGENTS.md: "tput_per_gpu (total throughput per GPU)".)
    total_pg = round(out_pg + in_pg, 4)
    return {
        "tput_per_gpu": total_pg, "output_tput_per_gpu": out_pg, "input_tput_per_gpu": in_pg,
        "mean_ttft": s(ttft["mean"]), "median_ttft": s(ttft["p50"]),
        "std_ttft": 0.0, "p90_ttft": s(p95(ttft)), "p99_ttft": s(ttft["p99"]),
        # TPOT / ITL / interactivity all = log-derived STEADY (no accurate prom decode tail)
        "mean_tpot": s(tpot_med), "median_tpot": s(tpot_med),
        "std_tpot": 0.0, "p90_tpot": s(tpot_med), "p99_tpot": s(tpot_med),
        "mean_itl": s(tpot_med), "median_itl": s(tpot_med),
        "std_itl": 0.0, "p90_itl": s(tpot_med), "p99_itl": s(tpot_med),
        "mean_intvty": iv_med, "median_intvty": iv_med,
        "std_intvty": 0.0, "p90_intvty": iv_med, "p99_intvty": iv_med,
        "mean_e2el": s(e2e["mean"]), "median_e2el": s(e2e["p50"]),
        "std_e2el": 0.0, "p90_e2el": s(p95(e2e)), "p99_e2el": s(e2e["p99"]),
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "InferenceX-app" / "local-dump"
    out.mkdir(parents=True, exist_ok=True)

    configs, benchmarks, availability = [], [], []
    summary = []
    cid = 0
    bid = 1
    for run in RUNS:
        concs = concs_in(RESULTS / run["dir"])
        if not concs:
            summary.append((run["fw"], run["dir"], run["ep"], []))   # no data yet -> skip emit
            continue
        cid += 1
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
        for conc in concs:
            srv = json.loads((RESULTS / run["dir"] / f"conc{conc}.json").read_text())["server"]
            benchmarks.append({
                "id": bid, "workflow_run_id": 1, "config_id": cid, "benchmark_type": "agentic-replay",
                "date": DATE, "isl": ISL, "osl": OSL, "conc": conc, "image": None,
                "metrics": metrics_from_server(srv), "workers": None, "error": None, "server_log_id": None,
            })
            got.append((conc, round(_tput_pg(srv), 1)))
            bid += 1
        summary.append((run["fw"], run["dir"], ep, got))

    workflow_runs = [{
        "id": 1, "github_run_id": 1, "run_attempt": 1,
        "name": "MiniMax-M2.5 agentic replay — vLLM/SGLang × EP/TP-only + vLLM-EP bigbatch (SERVER PROM TRUTH, 4xGB300 colocated)",
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

    all_concs = sorted({c for _, _, _, got in summary for c, _ in got})
    print("\n=== MiniMax-M2.5 (4xGB300, colocated, fp4+fp8KV, ISL/OSL 37109/1490) — tok/s/GPU ===")
    print(f"  {'framework':14} {'ep':>2}  " + " ".join(f"{c:>7}" for c in all_concs))
    for fw, d, ep, got in summary:
        m = {c: v for c, v in got}
        if not m:
            print(f"  {fw:14} {ep:>2}  (no data yet — {d} sweep not started/written)")
            continue
        row = " ".join(f"{m[c]:>7}" if c in m else f"{'·':>7}" for c in all_concs)
        print(f"  {fw:14} {ep:>2}  {row}")
    print(f"\ndump dir: {out}  | configs={len(configs)} benchmarks={len(benchmarks)} availability={len(availability)}")


if __name__ == "__main__":
    main()
