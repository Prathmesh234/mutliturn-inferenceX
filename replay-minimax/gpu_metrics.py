#!/usr/bin/env python3
"""Summarize a GPU-telemetry CSV produced during the measured window.

The serve script samples, once per second while the sweep runs:
  nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,\
temperature.gpu,memory.used --format=csv,noheader,nounits -l 1 > gpu.csv

This reduces that to per-GPU and deployment-wide hardware metrics — the physical
side of the throughput story (energy + utilization), complementing the
Prometheus serving metrics. First principles: throughput-per-Watt and GPU
utilization tell you whether the bottleneck is compute or memory/coordination.

Usage:  python gpu_metrics.py gpu.csv [out.json]
"""
import json
import sys
from collections import defaultdict


def summarize(csv_path):
    per = defaultdict(lambda: {"power": [], "util": [], "temp": [], "mem": []})
    n = 0
    with open(csv_path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                idx = int(parts[1])
                power, util, temp, mem = (float(parts[2]), float(parts[3]),
                                          float(parts[4]), float(parts[5]))
            except ValueError:
                continue  # header / malformed / "[N/A]"
            per[idx]["power"].append(power)
            per[idx]["util"].append(util)
            per[idx]["temp"].append(temp)
            per[idx]["mem"].append(mem)
            n += 1
    if not per:
        return {"error": "no GPU samples parsed", "samples": 0}

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    gpus = {}
    for idx, d in sorted(per.items()):
        gpus[idx] = {
            "avg_power_w": mean(d["power"]), "peak_power_w": round(max(d["power"]), 2),
            "avg_util_pct": mean(d["util"]),
            "avg_temp_c": mean(d["temp"]), "peak_temp_c": round(max(d["temp"]), 2),
            "avg_mem_used_mb": mean(d["mem"]),
        }
    n_gpu = len(gpus)
    all_power = [v["avg_power_w"] for v in gpus.values()]
    all_util = [v["avg_util_pct"] for v in gpus.values()]
    return {
        "n_gpu": n_gpu, "samples": n,
        # deployment-wide (mean across GPUs)
        "avg_power_w_per_gpu": round(sum(all_power) / n_gpu, 2),
        "total_avg_power_w": round(sum(all_power), 2),
        "avg_util_pct": round(sum(all_util) / n_gpu, 2),
        "peak_power_w_per_gpu": round(max(v["peak_power_w"] for v in gpus.values()), 2),
        "per_gpu": gpus,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    out = summarize(sys.argv[1])
    js = json.dumps(out, indent=2)
    print(js)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(js)
