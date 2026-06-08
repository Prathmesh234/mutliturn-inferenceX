#!/usr/bin/env python3
"""Build an InferenceX-app JSON dump from our replay sweeps (colocated + disagg).

The InferenceX dashboard can run with no database by reading a "dump dir" of
per-table JSON files (see packages/db/src/json-provider.ts; enabled by
DUMP_DIR in .env). This script emits that dump so the dashboard renders OUR
agentic-replay pareto data WITHOUT touching any frontend component.

  configs.json           9 configs (see RUNS below)
  workflow_runs.json     1 synthetic run (conclusion=success)
  benchmark_results.json 39 rows (colocated: 4 cfg x conc{4,8,16,32,64,128};
                                   disagg:    5 cfg x conc{4,8,16})
  availability.json      6 rows (model|hw|framework|precision|isl|osl|date) so
                         the model/sequence pickers list every series
  run_stats.json / eval_results.json / changelog_entries.json  empty

Series separation (no component changes): the dashboard groups rooflines by
hardware+framework. We map serving mode onto the framework field, which already
has registered keys in @semianalysisai/inferencex-constants FW_REGISTRY:

  colocated vLLM   -> framework "vllm"           (label "vLLM")
  colocated SGLang -> framework "sglang"         (label "SGLang")
  disagg    vLLM   -> framework "dynamo-vllm"    (label "Dynamo vLLM",  disagg=true)
  disagg    SGLang -> framework "dynamo-sglang"  (label "Dynamo SGLang", disagg=true)

=> per model, up to 4 distinct rooflines/colours = engine x {colocated,disagg}.
The 3 disagg-vLLM topologies (1P1D/2P1D/1P2D) share framework "dynamo-vllm", so
their points fall under ONE "Dynamo vLLM" frontier (the dashboard has no
per-topology roofline grouping); likewise the 2 disagg-SGLang topologies. Each
point keeps its own prefill/decode GPU split in the row for the tooltip.

Units / normalisation (proven by the app + matched to the prior dump):
  * Latency fields (*_ttft/_tpot/_itl/_e2el) are SECONDS. Source conc*.json is
    MILLISECONDS -> divide by 1000.
  * Interactivity (tok/s/user) = 1000 / tpot_ms = 1 / tpot_s.
  * Throughput is PER-GPU over the whole deployment: output_tok_s / total_gpu,
    where total_gpu = num_prefill_gpu + num_decode_gpu (colocated TP4 => 4;
    disagg 4p_4d => 8, 8p_4d => 12, 4p_8d => 12).
  * ISL/OSL set to the NOMINAL bench bucket 8192/1024 ("8K/1K"), one of the only
    three sequence buckets InferenceX registers (1k1k, 1k8k, 8k1k); any other
    value makes islOslToSequence() return null and the row is dropped. The real
    agentic workload is variable multi-turn (recorded ISL median ~5-6k, OSL
    ~660-950) -- only the sequence LABEL is bucketed; plotted metrics are real.

Usage:  python make_inferencex_dump.py [out_dir]
        (default out_dir: ~/InferenceX-app/local-dump)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path("/mnt/home/ppbhatt500/gpumode-triton/results")
DATE = "2026-06-08"
ISL, OSL = 8192, 1024

COLOCATED_CONC = (4, 8, 16, 32, 64, 128)
DISAGG_CONC = (4, 8, 16)

# Each run: result dir + the config it represents. num_prefill_gpu / num_decode_gpu
# are derived as tp * num_workers (ep=1 for all our runs). total_gpu (the per-GPU
# divisor) = num_prefill_gpu + num_decode_gpu.
#   dir, model, framework, disagg, multinode, p_tp, p_w, d_tp, d_w, conc
RUNS = [
    # --- colocated (single GB300 node, TP4, prefill==decode==same 4 GPUs) ---
    dict(dir="dsv4_b1",        model="dsv4",     fw="vllm",          disagg=False, mn=False, p_tp=4, p_w=1, d_tp=4, d_w=1, conc=COLOCATED_CONC),
    dict(dir="dsv4_b1_sglang", model="dsv4",     fw="sglang",        disagg=False, mn=False, p_tp=4, p_w=1, d_tp=4, d_w=1, conc=COLOCATED_CONC),
    dict(dir="kimi_b1",        model="kimik2.5", fw="vllm",          disagg=False, mn=False, p_tp=4, p_w=1, d_tp=4, d_w=1, conc=COLOCATED_CONC),
    dict(dir="kimi_b1_sglang", model="kimik2.5", fw="sglang",        disagg=False, mn=False, p_tp=4, p_w=1, d_tp=4, d_w=1, conc=COLOCATED_CONC),
    # --- disaggregated (Dynamo P/D split, multinode, dsv4 only) ---
    # 1P1D = 1 prefill node (TP4) + 1 decode node (TP4)  -> 4 + 4  =  8 GPU
    dict(dir="dsv4_vllm_4p_4d",   model="dsv4", fw="dynamo-vllm",   disagg=True, mn=True, p_tp=4, p_w=1, d_tp=4, d_w=1, conc=DISAGG_CONC),
    # 2P1D = 2 prefill nodes (TP4 x2) + 1 decode node       -> 8 + 4  = 12 GPU
    dict(dir="dsv4_vllm_8p_4d",   model="dsv4", fw="dynamo-vllm",   disagg=True, mn=True, p_tp=4, p_w=2, d_tp=4, d_w=1, conc=DISAGG_CONC),
    # 1P2D = 1 prefill node + 2 decode nodes (TP4 x2)       -> 4 + 8  = 12 GPU
    dict(dir="dsv4_vllm_4p_8d",   model="dsv4", fw="dynamo-vllm",   disagg=True, mn=True, p_tp=4, p_w=1, d_tp=4, d_w=2, conc=DISAGG_CONC),
    # sglang 1P1D                                            -> 4 + 4  =  8 GPU
    dict(dir="dsv4_sglang_4p_4d", model="dsv4", fw="dynamo-sglang", disagg=True, mn=True, p_tp=4, p_w=1, d_tp=4, d_w=1, conc=DISAGG_CONC),
    # sglang 1P2D                                            -> 4 + 8  = 12 GPU
    dict(dir="dsv4_sglang_4p_8d", model="dsv4", fw="dynamo-sglang", disagg=True, mn=True, p_tp=4, p_w=1, d_tp=4, d_w=2, conc=DISAGG_CONC),
    # NOTE: dsv4_sglang_8p_4d is intentionally absent -- that run failed at the
    # MoE "Hidden size mismatch" and produced no results.
]


def num_gpu(run):
    npg = run["p_tp"] * run["p_w"]
    ndg = run["d_tp"] * run["d_w"]
    return npg, ndg


def cfg(cid, run):
    npg, ndg = num_gpu(run)
    return {
        "id": cid, "hardware": "gb300", "framework": run["fw"], "model": run["model"],
        "precision": "fp4", "spec_method": "none",
        "disagg": run["disagg"], "is_multinode": run["mn"],
        "prefill_tp": run["p_tp"], "prefill_ep": 1, "prefill_dp_attention": False,
        "prefill_num_workers": run["p_w"],
        "decode_tp": run["d_tp"], "decode_ep": 1, "decode_dp_attention": False,
        "decode_num_workers": run["d_w"],
        "num_prefill_gpu": npg, "num_decode_gpu": ndg,
    }


def metrics_from(c, total_gpu):
    out_tps = c["output_throughput_tok_per_s"]
    tot_tps = c["total_token_throughput_tok_per_s"]
    in_tps = max(tot_tps - out_tps, 0.0)
    ttft_ms, tpot_ms, e2e_ms = c["ttft_ms"], c["tpot_ms"], c["e2e_ms"]
    itl_ms = c.get("itl_ms", {})

    def to_s(d):  # ms -> s for every stat key present
        return {k: round(v / 1000.0, 6) for k, v in d.items()}

    ttft, tpot, e2e, itl = to_s(ttft_ms), to_s(tpot_ms), to_s(e2e_ms), to_s(itl_ms)

    def intvty(tpot_ms_val):  # tokens/s per user = 1000/tpot_ms = 1/tpot_s
        return round(1000.0 / tpot_ms_val, 4) if tpot_ms_val else 0.0

    return {
        "tput_per_gpu": round(out_tps / total_gpu, 4),
        "output_tput_per_gpu": round(out_tps / total_gpu, 4),
        "input_tput_per_gpu": round(in_tps / total_gpu, 4),
        "mean_ttft": ttft.get("mean", 0.0), "median_ttft": ttft.get("p50", 0.0),
        "std_ttft": 0.0, "p99_ttft": ttft.get("p99", 0.0),
        "mean_tpot": tpot.get("mean", 0.0), "median_tpot": tpot.get("p50", 0.0),
        "std_tpot": 0.0, "p99_tpot": tpot.get("p99", 0.0),
        # intvty consumes the ORIGINAL milliseconds (1000/tpot_ms = 1/tpot_s).
        "mean_intvty": intvty(tpot_ms.get("mean", 0.0)),
        "median_intvty": intvty(tpot_ms.get("p50", 0.0)),
        "std_intvty": 0.0, "p99_intvty": intvty(tpot_ms.get("p99", 0.0)),
        "mean_itl": itl.get("mean", tpot.get("mean", 0.0)),
        "median_itl": tpot.get("p50", 0.0),
        "std_itl": 0.0, "p99_itl": itl.get("p99", tpot.get("p99", 0.0)),
        "mean_e2el": e2e.get("mean", 0.0), "median_e2el": e2e.get("p50", 0.0),
        "std_e2el": 0.0, "p99_e2el": e2e.get("p99", 0.0),
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "InferenceX-app" / "local-dump"
    out.mkdir(parents=True, exist_ok=True)

    configs, benchmarks = [], []
    avail_seen, availability = set(), []
    bid = 1
    for cid, run in enumerate(RUNS, start=1):
        configs.append(cfg(cid, run))
        npg, ndg = num_gpu(run)
        # Per-GPU divisor = physical deployment size. DISAGG: prefill and decode
        # run on SEPARATE GPUs -> npg + ndg. COLOCATED: prefill and decode share
        # the SAME single TP4 node -> just decode_tp (== 4), NOT 4+4. This matches
        # the dashboard, which sets tp = num_prefill_gpu + num_decode_gpu only when
        # disagg, else tp = decode_tp (see data-transforms.md step 2).
        total_gpu = (npg + ndg) if run["disagg"] else (run["d_tp"] * run.get("d_ep", 1))

        akey = (run["model"], run["fw"])
        if akey not in avail_seen:
            avail_seen.add(akey)
            availability.append({
                "model": run["model"], "isl": ISL, "osl": OSL, "precision": "fp4",
                "hardware": "gb300", "framework": run["fw"], "spec_method": "none",
                "disagg": run["disagg"], "date": DATE,
            })

        for conc in run["conc"]:
            f = RESULTS / run["dir"] / f"conc{conc}.json"
            if not f.exists():
                print(f"  WARN missing {f}", file=sys.stderr)
                continue
            c = json.loads(f.read_text())
            benchmarks.append({
                "id": bid, "workflow_run_id": 1, "config_id": cid,
                "benchmark_type": "agentic-replay", "date": DATE,
                "isl": ISL, "osl": OSL, "conc": conc, "image": None,
                "metrics": metrics_from(c, total_gpu), "workers": None,
                "error": None, "server_log_id": None,
            })
            bid += 1

    workflow_runs = [{
        "id": 1, "github_run_id": 1, "run_attempt": 1,
        "name": "Agentic replay pareto - DSv4/Kimi x vLLM/SGLang, colocated + Dynamo disagg (GB300)",
        "status": "completed", "conclusion": "success",
        "head_sha": "local", "head_branch": "custom-replay", "html_url": None,
        "created_at": f"{DATE}T00:00:00.000Z",
        "run_started_at": f"{DATE}T00:00:00.000Z", "date": DATE,
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
    print(f"\ndump dir: {out}")
    print(f"  configs={len(configs)} benchmarks={len(benchmarks)} availability={len(availability)}")


if __name__ == "__main__":
    main()
