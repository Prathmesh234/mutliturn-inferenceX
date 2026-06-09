#!/usr/bin/env python3
"""Sweep concurrency and trace the throughput-vs-(interactivity/TTFT/E2E) pareto.

Runs replay_bench.py once per concurrency point (subprocess, clean isolation),
reads the SERVER PROMETHEUS metrics from each result, and emits:
  <out>/conc<N>.json   per-point
  <out>/pareto.csv     one row per concurrency (server-truth)
  <out>/pareto.png     throughput/GPU vs interactivity, vs p99 TTFT, vs p99 E2E

First principles: as concurrency rises, throughput/GPU climbs while
interactivity (tok/s/user) and the p99 latencies degrade. The frontier is the
deployment's throughput-at-a-latency-SLA identity. We also surface the stage-2
capacity metrics (prefix-cache hit rate, GPU/CPU KV usage) per point.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "replay_bench.py"


def run_point(conc, args):
    out = Path(args.result_dir) / f"conc{conc}.json"
    cmd = [sys.executable, str(BENCH),
           "--dataset", args.dataset, "--base-url", args.base_url,
           "--metrics-url", args.metrics_url, "--model", args.model,
           "--concurrency", str(conc), "--n-gpu", str(args.n_gpu),
           "--prearrange-frac", str(args.prearrange_frac),
           "--ramp-seconds", str(args.ramp_seconds),
           "--result-dir", args.result_dir, "--result-filename", f"conc{conc}.json"]
    if args.extra_body:
        cmd += ["--extra-body", args.extra_body]
    print(f"\n===== concurrency {conc} =====", flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text())


COLS = ["concurrency", "output_tput_per_gpu", "intvty_p50", "intvty_p99",
        "ttft_p50_ms", "ttft_p99_ms", "e2e_p50_ms", "e2e_p99_ms",
        "gpu_prefix_cache_hit_rate", "gpu_kv_usage_perc", "cpu_kv_usage_perc",
        "num_running", "num_waiting", "completed_turns"]


def row_of(r):
    s = r.get("server", {}) or {}
    ttft = s.get("ttft_ms") or {}
    e2e = s.get("e2e_ms") or {}
    return {
        "concurrency": r["concurrency"],
        "output_tput_per_gpu": s.get("output_tput_per_gpu"),
        "intvty_p50": s.get("intvty_p50"), "intvty_p99": s.get("intvty_p99"),
        "ttft_p50_ms": ttft.get("p50"), "ttft_p99_ms": ttft.get("p99"),
        "e2e_p50_ms": e2e.get("p50"), "e2e_p99_ms": e2e.get("p99"),
        "gpu_prefix_cache_hit_rate": s.get("gpu_prefix_cache_hit_rate"),
        "gpu_kv_usage_perc": s.get("gpu_kv_usage_perc"),
        "cpu_kv_usage_perc": s.get("cpu_kv_usage_perc"),
        "num_running": s.get("num_running"), "num_waiting": s.get("num_waiting"),
        "completed_turns": (r.get("client") or {}).get("completed_turns"),
    }


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(row_of(r))


def plot(rows, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}", file=sys.stderr)
        return
    rows = sorted(rows, key=lambda r: r["concurrency"])
    tput = [row_of(r)["output_tput_per_gpu"] for r in rows]
    panels = [("intvty_p99", "Interactivity p99 (tok/s/user)"),
              ("ttft_p99_ms", "TTFT p99 (ms)"), ("e2e_p99_ms", "E2E p99 (ms)")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, (key, lab) in zip(ax, panels):
        x = [row_of(r)[key] for r in rows]
        a.plot(x, tput, "-o", color="#2a6fc2", lw=2)
        for xx, yy, r in zip(x, tput, rows):
            if xx is not None and yy is not None:
                a.annotate(f"c{r['concurrency']}", (xx, yy), textcoords="offset points", xytext=(6, 4), fontsize=8)
        a.set_xlabel(lab); a.set_ylabel("Output throughput / GPU (tok/s)")
        a.grid(True, alpha=0.3); a.set_title("throughput vs " + lab)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--model", default="minimax")
    ap.add_argument("--n-gpu", type=int, default=4)
    ap.add_argument("--concurrencies", default="4,8,16,32,64,128")
    ap.add_argument("--prearrange-frac", type=float, default=0.75)
    ap.add_argument("--ramp-seconds", type=float, default=60.0)
    ap.add_argument("--extra-body", default=None)
    ap.add_argument("--result-dir", default="out")
    ap.add_argument("--title", default=None)
    ap.add_argument("--replot", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    if args.replot:
        rows = [json.loads(p.read_text()) for p in sorted(Path(args.result_dir).glob("conc*.json"),
                key=lambda q: int(q.stem[4:]))]
    else:
        rows = []
        for c in [int(x) for x in args.concurrencies.split(",") if x.strip()]:
            try:
                rows.append(run_point(c, args))
            except subprocess.CalledProcessError as e:
                print(f"conc {c} failed: {e}", file=sys.stderr)
    if not rows:
        print("no successful points", file=sys.stderr); sys.exit(1)
    write_csv(rows, Path(args.result_dir) / "pareto.csv")
    print(f"wrote {Path(args.result_dir) / 'pareto.csv'}")
    plot(rows, Path(args.result_dir) / "pareto.png", args.title or f"MiniMax-M2.5 {args.model}")
    print("\nconc | tput/gpu | intvty_p99 | ttft_p99 | e2e_p99 | cache_hit | kv_use")
    for r in sorted(rows, key=lambda r: r["concurrency"]):
        x = row_of(r)
        print(f"{x['concurrency']:>4} | {str(x['output_tput_per_gpu']):>8} | "
              f"{str(x['intvty_p99']):>10} | {str(x['ttft_p99_ms']):>8} | "
              f"{str(x['e2e_p99_ms']):>7} | {str(x['gpu_prefix_cache_hit_rate']):>9} | "
              f"{str(x['gpu_kv_usage_perc']):>6}")


if __name__ == "__main__":
    main()
