#!/usr/bin/env python3
"""p95-only pareto charts with a 95% CI box (TTFT, TPOT, Throughput) top-right.

Per request: charts show ONLY p95 (no p50/p99). TTFT-p95 and TPOT-p95 vs output
throughput, one curve per result dir. p95 source, "whichever is more accurate":
  - MEASURED client-side p95 (conc<N>.json ttft_ms.p95) when present (exact).
  - else INTERPOLATED from the measured percentiles (log-linear: ttft p90->p99,
    tpot/e2e p50->p99) and the chart is labelled "(p95 estimated)".

Top-right CI box: 95% confidence interval of the MEAN across the concurrency
points on the chart, for TTFT-p95, TPOT-p95 and Throughput, using
  mean ± t(.975, n-1) * s/sqrt(n)
(parametric; t critical value for small n). Shown as [lo, hi].

Usage:  python plot_p95_ci.py <result_dir> [<result_dir> ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# t critical values for two-sided 95% (df -> t); fall back to z=1.96 for df>30.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def tcrit(n: int) -> float:
    df = n - 1
    if df <= 0:
        return float("nan")
    if df in T95:
        return T95[df]
    return 1.96 if df > 30 else T95[min(T95, key=lambda k: abs(k - df))]


def ci_mean(vals):
    """95% CI of the mean -> (lo, hi, measured?)."""
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    mu = sum(vals) / n
    s = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
    se = s / math.sqrt(n)
    h = tcrit(n) * se
    return (mu - h, mu + h)


def loglin(q_lo, v_lo, q_hi, v_hi, qt=95):
    frac = (qt - q_lo) / (q_hi - q_lo)
    if v_lo > 0 and v_hi > 0:
        return math.exp(math.log(v_lo) + frac * (math.log(v_hi) - math.log(v_lo)))
    return v_lo + frac * (v_hi - v_lo)


def p95_of(block: dict):
    """(value, measured?) for a latency block."""
    if block.get("p95") is not None:
        return block["p95"], True
    if "p90" in block and "p99" in block:          # ttft: tight bracket
        return loglin(90, block["p90"], 99, block["p99"]), False
    if "p50" in block and "p99" in block:          # tpot/e2e: wide bracket
        return loglin(50, block["p50"], 99, block["p99"]), False
    return block.get("p99", 0.0), False


def load(d: Path):
    rows = []
    for p in sorted(d.glob("conc*.json"),
                    key=lambda q: int(q.stem.replace("conc", ""))):
        if p.name.endswith(".old6777"):
            continue
        rows.append(json.loads(p.read_text()))
    return rows


def fmt_ci(lo, hi, unit):
    if math.isnan(lo):
        return "[n/a]"
    lo = max(0.0, lo)   # latency/throughput can't be negative
    return f"[{lo:,.1f}, {hi:,.1f}] {unit}"


def plot_dir(d: Path):
    rows = load(d)
    if len(rows) < 1:
        print(f"  {d.name}: no conc*.json", file=sys.stderr)
        return
    rows.sort(key=lambda r: r["concurrency"])
    conc = [r["concurrency"] for r in rows]
    thr = [r["output_throughput_tok_per_s"] for r in rows]
    ttft, t_meas = zip(*[p95_of(r["ttft_ms"]) for r in rows])
    tpot, p_meas = zip(*[p95_of(r["tpot_ms"]) for r in rows])
    measured = all(t_meas) and all(p_meas)

    # 95% CIs of the mean across points
    ci_t = ci_mean(list(ttft)); ci_p = ci_mean(list(tpot)); ci_x = ci_mean(thr)
    box = (f"95% CI of mean (n={len(rows)} pts)\n"
           f"TTFT p95: {fmt_ci(*ci_t, 'ms')}\n"
           f"TPOT p95: {fmt_ci(*ci_p, 'ms')}\n"
           f"Tput:     {fmt_ci(*ci_x, 'tok/s')}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for a, y, lab in zip(ax, (ttft, tpot), ("TTFT p95 (ms)", "TPOT p95 (ms)")):
        a.plot(thr, y, "-o", color="#c25a3a", lw=2, markersize=6, label="p95")
        for x, yy, c in zip(thr, y, conc):
            a.annotate(f"c{c}", (x, yy), textcoords="offset points",
                       xytext=(6, 4), fontsize=8)
        a.set_xlabel("Output throughput (tok/s)")
        a.set_ylabel(lab)
        a.grid(True, alpha=0.3)
        a.set_title(lab + " vs throughput")
        a.legend(fontsize=8, loc="lower right")
    # CI box in the TOP-RIGHT of the right-hand panel
    ax[1].text(0.97, 0.97, box, transform=ax[1].transAxes, ha="right", va="top",
               fontsize=8.5, family="monospace",
               bbox=dict(boxstyle="round", fc="#fff7f2", ec="#c25a3a", alpha=0.95))
    note = "p95 measured (client-side)" if measured else "p95 ESTIMATED (interp from measured percentiles)"
    fig.suptitle(f"{d.name} — p95 pareto · {note}")
    fig.tight_layout()
    out = d / "pareto_p95.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}  ({'measured' if measured else 'estimated'} p95, n={len(rows)})")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for d in sys.argv[1:]:
        plot_dir(Path(d))


if __name__ == "__main__":
    main()
