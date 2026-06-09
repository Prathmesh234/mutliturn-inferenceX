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


COLS = ["concurrency", "output_tput_per_gpu_steady",
        "intvty_steady", "tpot_steady_ms",
        "ttft_p50_ms", "ttft_p99_ms", "e2e_p50_ms", "e2e_p99_ms",
        "gpu_prefix_cache_hit_rate", "gpu_kv_usage_perc", "cpu_kv_usage_perc",
        "num_running", "num_waiting", "completed_turns"]


def tput_pg(r):
    """Plotted-frontier throughput = STEADY-STATE full-batch tput/GPU (recompute_steady.py:
    median decode tput over engine-log samples with running >= 0.9*conc, per GPU). This is
    the canonical throughput in conc<N>.json; recompute_steady removes the whole-window
    average entirely. The `.get(output_tput_per_gpu)` fallback only fires for a live sweep
    plotted BEFORE recompute_steady has run (the window avg is diluted up to ~2.2x at high
    conc by ramp/drain on the 150-session set, so it's a placeholder only)."""
    s = r.get("server", {}) or {}
    return s.get("output_tput_per_gpu_steady", s.get("output_tput_per_gpu"))


def row_of(r):
    s = r.get("server", {}) or {}
    ttft = s.get("ttft_ms") or {}
    e2e = s.get("e2e_ms") or {}
    return {
        "concurrency": r["concurrency"],
        "output_tput_per_gpu_steady": s.get("output_tput_per_gpu_steady"),
        "intvty_steady": s.get("intvty_steady"),
        "tpot_steady_ms": s.get("tpot_steady_ms"),
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


def _mono_cubic(xs, ys, n=240):
    """Fritsch-Carlson monotone cubic Hermite interpolation over a strictly
    increasing xs. Shape-preserving: never overshoots or invents bumps between
    data points, so the smoothed pareto curve stays faithful to the measurement."""
    import numpy as np
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    h = np.diff(xs); delta = np.diff(ys) / h
    m = np.empty_like(xs)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(len(delta)):                       # enforce monotone tangents
        if delta[i] == 0:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                t = 3.0 / s ** 0.5
                m[i], m[i + 1] = t * a * delta[i], t * b * delta[i]
    xg = np.linspace(xs[0], xs[-1], n)
    idx = np.clip(np.searchsorted(xs, xg) - 1, 0, len(xs) - 2)
    t = (xg - xs[idx]) / h[idx]
    h00 = (1 + 2 * t) * (1 - t) ** 2; h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t); h11 = t * t * (t - 1)
    yg = h00 * ys[idx] + h10 * h[idx] * m[idx] + h01 * ys[idx + 1] + h11 * h[idx] * m[idx + 1]
    return xg, yg


def _smooth_xy(xvals, yvals):
    """Smooth curve through (metric, throughput) points. Throughput is the
    monotone axis (rises with concurrency), so we interpolate metric=f(tput)
    and trace it back as (metric_smooth, tput_grid) -> a clean elbow, no zigzag."""
    import numpy as np
    pts = [(x, y) for x, y in zip(xvals, yvals) if x is not None and y is not None]
    if len(pts) < 3:
        return None
    pts.sort(key=lambda p: p[1])                       # sort by throughput (y)
    ys = [p[1] for p in pts]; xs = [p[0] for p in pts]
    ys = list(ys)                                      # ensure strictly increasing y
    for i in range(1, len(ys)):
        if ys[i] <= ys[i - 1]:
            ys[i] = ys[i - 1] + 1e-6
    tg, xg = _mono_cubic(ys, xs)                       # x(metric) = f(tput)
    return np.asarray(xg), np.asarray(tg)


def plot(rows, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}", file=sys.stderr)
        return
    rows = sorted(rows, key=lambda r: r["concurrency"])
    # Drop c4 from the plotted frontier: it runs first against a cold radix cache
    # (prearrange only warms `concurrency` sessions, not the shared prefix), so its
    # TTFT/interactivity are a warm-up artifact, not steady-state. Kept in pareto.csv.
    rows = [r for r in rows if r["concurrency"] != 4]
    tput = [tput_pg(r) for r in rows]
    panels = [("intvty_steady", "Interactivity (steady, tok/s/user)"),
              ("ttft_p99_ms", "TTFT p99 (ms)"), ("e2e_p99_ms", "E2E p99 (ms)")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, (key, lab) in zip(ax, panels):
        x = [row_of(r)[key] for r in rows]
        sm = _smooth_xy(x, tput)
        if sm is not None:                              # smooth monotone-cubic frontier
            a.plot(sm[0], sm[1], "-", color="#2a6fc2", lw=2, zorder=2)
            a.plot([xx for xx in x if xx is not None],
                   [yy for xx, yy in zip(x, tput) if xx is not None],
                   "o", color="#2a6fc2", ms=6, zorder=3)
        else:
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
    print("\nconc | tput/gpu(steady) | intvty(steady) | tpot_steady_ms | ttft_p99 | e2e_p99 | cache_hit")
    for r in sorted(rows, key=lambda r: r["concurrency"]):
        x = row_of(r)
        print(f"{x['concurrency']:>4} | {str(tput_pg(r)):>16} | {str(x['intvty_steady']):>14} | "
              f"{str(x['tpot_steady_ms']):>14} | {str(x['ttft_p99_ms']):>8} | "
              f"{str(x['e2e_p99_ms']):>7} | {str(x['gpu_prefix_cache_hit_rate']):>9}")


if __name__ == "__main__":
    main()
