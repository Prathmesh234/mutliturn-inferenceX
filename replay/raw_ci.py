#!/usr/bin/env python3
"""Measured bootstrap CIs from raw_conc<c>.jsonl files (written by replay_bench.py).

Each raw file has one JSON row per completed request: ttft_ms, tpot_ms, e2e_ms, isl, osl.
This computes a percentile bootstrap CI for the MEAN and for p50/p90/p99 of TTFT/TPOT/E2E
— real intervals, no distributional assumption.

Usage:
  python raw_ci.py results/dsv4_vllm_4p_4d            # all raw_conc*.jsonl in a dir
  python raw_ci.py results/dsv4_vllm_4p_4d/raw_conc8.jsonl
"""
import sys, json, glob, os
import numpy as np

B = 10000          # bootstrap resamples
SEED = 0           # fixed for reproducibility (no Date/random-seed drift)

def boot_ci(x, stat, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = stat(x[idx], axis=1)
    lo, hi = np.percentile(bs, [(1-level)/2*100, (1+level)/2*100])
    return float(stat(x)), float(lo), float(hi)

def report(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    print(f"\n== {path}  (n={n} requests) ==")
    for field in ("ttft_ms", "tpot_ms", "e2e_ms"):
        vals = [r[field] for r in rows if r.get(field) is not None]
        if not vals:
            continue
        m, mlo, mhi = boot_ci(vals, lambda a, axis=None: np.mean(a, axis=axis))
        line = f"  {field:8s} n={len(vals):4d}  mean={m:8.1f}  95% CI [{mlo:8.1f}, {mhi:8.1f}]"
        for q in (50, 90, 99):
            pv, plo, phi = boot_ci(vals, lambda a, axis=None: np.percentile(a, q, axis=axis))
            line += f"\n              p{q:<2d}={pv:8.1f}  95% CI [{plo:8.1f}, {phi:8.1f}]"
        print(line)

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    arg = sys.argv[1]
    paths = ([arg] if arg.endswith(".jsonl")
             else sorted(glob.glob(os.path.join(arg, "raw_conc*.jsonl"))))
    if not paths:
        print(f"no raw_conc*.jsonl found at {arg}"); sys.exit(1)
    for p in paths:
        report(p)

if __name__ == "__main__":
    main()
