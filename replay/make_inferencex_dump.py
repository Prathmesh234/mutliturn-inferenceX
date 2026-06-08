#!/usr/bin/env python3
"""Build a minimal InferenceX-app JSON dump from our 4 colocated replay sweeps.

The InferenceX dashboard can run with no database by reading a "dump dir" of
per-table JSON files (see packages/db/src/json-provider.ts). This script emits
just enough of that dump to render OUR colocated agentic-replay pareto data:

  configs.json           4 configs: {dsv4,kimik2.5} x {vllm,sglang}, gb300, fp4, colocated TP4
  workflow_runs.json     1 synthetic run (conclusion=success)
  benchmark_results.json 4 configs x conc {4,8,16} = 12 rows, metrics from conc*.json
  availability.json      4 rows (model|hw|fw|precision|isl|osl|date) so the picker lists them
  run_stats.json / eval_results.json / changelog_entries.json  empty

Metric keys match benchmark-transform.ts (all read with `?? 0`). Throughput is
divided by the colocated GPU count (TP4 = 4). ISL/OSL are set to a NOMINAL
per-model median so vLLM and SGLang overlay on the same sequence for comparison.

Usage:  python make_inferencex_dump.py [out_dir]
        (default out_dir: ~/InferenceX-app/local-dump)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RESULTS = Path("/mnt/home/ppbhatt500/gpumode-triton/results")
DATE = "2026-06-08"
GPUS = 4  # colocated TP4

# (result_dir, model_key, framework, nominal_isl, nominal_osl)
RUNS = [
    ("dsv4_b1",        "dsv4",     "vllm",   5220, 671),
    ("dsv4_b1_sglang", "dsv4",     "sglang", 5220, 671),
    ("kimi_b1",        "kimik2.5", "vllm",   5760, 768),
    ("kimi_b1_sglang", "kimik2.5", "sglang", 5760, 768),
]


def cfg(cid, model, framework):
    return {
        "id": cid, "hardware": "gb300", "framework": framework, "model": model,
        "precision": "fp4", "spec_method": "none", "disagg": False,
        "is_multinode": False,
        "prefill_tp": GPUS, "prefill_ep": 1, "prefill_dp_attention": False,
        "prefill_num_workers": 1,
        "decode_tp": GPUS, "decode_ep": 1, "decode_dp_attention": False,
        "decode_num_workers": 1,
        "num_prefill_gpu": GPUS, "num_decode_gpu": GPUS,
    }


def metrics_from(c):
    out_tps = c["output_throughput_tok_per_s"]
    tot_tps = c["total_token_throughput_tok_per_s"]
    in_tps = max(tot_tps - out_tps, 0.0)
    ttft, tpot, e2e = c["ttft_ms"], c["tpot_ms"], c["e2e_ms"]
    itl = c.get("itl_ms", {})

    def intvty(tpot_ms):  # tokens/s per user
        return round(1000.0 / tpot_ms, 4) if tpot_ms else 0.0

    return {
        "tput_per_gpu": round(out_tps / GPUS, 4),
        "output_tput_per_gpu": round(out_tps / GPUS, 4),
        "input_tput_per_gpu": round(in_tps / GPUS, 4),
        "mean_ttft": ttft.get("mean", 0.0), "median_ttft": ttft.get("p50", 0.0),
        "std_ttft": 0.0, "p99_ttft": ttft.get("p99", 0.0),
        "mean_tpot": tpot.get("mean", 0.0), "median_tpot": tpot.get("p50", 0.0),
        "std_tpot": 0.0, "p99_tpot": tpot.get("p99", 0.0),
        "mean_intvty": intvty(tpot.get("mean", 0.0)),
        "median_intvty": intvty(tpot.get("p50", 0.0)),
        "std_intvty": 0.0, "p99_intvty": intvty(tpot.get("p99", 0.0)),
        "mean_itl": itl.get("mean", tpot.get("mean", 0.0)),
        "median_itl": tpot.get("p50", 0.0),
        "std_itl": 0.0, "p99_itl": itl.get("p99", tpot.get("p99", 0.0)),
        "mean_e2el": e2e.get("mean", 0.0), "median_e2el": e2e.get("p50", 0.0),
        "std_e2el": 0.0, "p99_e2el": e2e.get("p99", 0.0),
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "InferenceX-app" / "local-dump"
    out.mkdir(parents=True, exist_ok=True)

    configs, benchmarks, availability = [], [], []
    bid = 1
    for cid, (rdir, model, fw, isl, osl) in enumerate(RUNS, start=1):
        configs.append(cfg(cid, model, fw))
        availability.append({
            "model": model, "isl": isl, "osl": osl, "precision": "fp4",
            "hardware": "gb300", "framework": fw, "spec_method": "none",
            "disagg": False, "date": DATE,
        })
        for conc in (4, 8, 16):
            f = RESULTS / rdir / f"conc{conc}.json"
            if not f.exists():
                print(f"  WARN missing {f}", file=sys.stderr)
                continue
            c = json.loads(f.read_text())
            benchmarks.append({
                "id": bid, "workflow_run_id": 1, "config_id": cid,
                "benchmark_type": "agentic-replay", "date": DATE,
                "isl": isl, "osl": osl, "conc": conc, "image": None,
                "metrics": metrics_from(c), "workers": None,
                "error": None, "server_log_id": None,
            })
            bid += 1

    workflow_runs = [{
        "id": 1, "github_run_id": 1, "run_attempt": 1,
        "name": "Colocated agentic replay (DSv4/Kimi x vLLM/SGLang, GB300)",
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


if __name__ == "__main__":
    main()
