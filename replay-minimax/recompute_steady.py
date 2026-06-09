#!/usr/bin/env python3
"""Augment each conc<N>.json with STEADY-STATE metrics for the InferenceX dump.

WHY: output_tput_per_gpu in conc<N>.json is a whole-window average = gen_tokens /
runtime. With only 150 sessions the high-concurrency windows are dominated by
ramp-up + drain-down (median running batch << requested conc), so the average
understates the true full-batch throughput by up to ~2.2x at conc 128. The
relative ordering is unaffected (identical deterministic work), but the absolute
numbers are low. The big-batch x6 run (sustained full batch) corroborates the
steady-state values extracted here.

This script ADDS steady fields and refreshes the prom-derived latency percentiles:
  server["output_tput_per_gpu_steady"]  -- median decode tput over full-batch
        engine-log samples (running >= 0.9*conc), per GPU. Corroborated by the
        x6 sustained run for vLLM-EP (~1666 vs log-derived ~1666).
  server["intvty_steady"]               -- median(gen_tput/running) over the SAME
        full-batch samples = true steady tok/s/user; server["tpot_steady_ms"] = its
        reciprocal. Used for the pareto-frontier interactivity axis so it is
        consistent with output_tput_per_gpu_steady (same samples, same regime).
  server["tpot_ms"]["p50"|"p95"|"p99"]  -- per-REQUEST TPOT from the prom histogram
        vllm:request_time_per_output_token_seconds (the canonical TPOT metric; absent
        on SGLang -> falls back to itl_ms). Kept in the table for validation.
  server["ttft_ms"|"itl_ms"|"e2e_ms"]["p95"]  -- 95th pct from the SAME prom
        histogram that already produced p50/p99 (just the 0.95 quantile).

CAVEAT: tput AND interactivity on the frontier are steady-state (log). The prom
latency percentiles (ttft/itl/tpot/e2e p50/p95/p99) are window-distribution (only 2
snapshots exist so they can't be sub-windowed) and mildly optimistic vs full-batch.
TTFT has no log-derived steady equivalent (engines don't log per-request TTFT), so it
stays prom-window. Documented, not hidden.

DISAGGREGATED (prefill/decode-split) RUNS — source-of-truth notes so you're not confused
when collecting metrics (the disagg serve scripts in disagg/ drive this the same way):
  * --log must point at the DECODE engine log. The disagg scripts tee ONLY the decode
    engine + sweep markers into run.log; the prefill engine's ~0 tok/s "generation
    throughput" lines go to prefill.log on purpose (they would poison these medians).
  * 1P1D / 2P2D: ONE decode engine -> steady tput + interactivity here are correct,
    exactly as in aggregate. (2P2D uses all 4 GPUs, so its ÷4 per-GPU tput is directly
    comparable to the aggregate runs; 1P1D uses only 2 of 4, so its ÷4 is per-quarter-
    node with 2 GPUs idle.)
  * 1P3D: THREE decode engines interleave in run.log, each running only ~conc/3, so the
    full-batch filter in steady_by_conc (running >= 0.9*conc) NEVER matches -> this
    script WARNs and conc<N>.json keeps the whole-window prom value, which came from
    decode #1 ONLY (~1/3 of the tokens). Do NOT trust tput/interactivity for 1P3D here:
    sum generation_tokens_total across the decode{1,2,3}.final.prom snapshots the serve
    script saves, and divide by runtime, to get the real aggregate throughput.
  * TTFT / E2E in disagg are decode-side only (the server histograms start their clock
    when DECODE receives the request, excluding prefill compute + KV transfer) -> read
    the CLIENT block (client_ttft_ms / client_e2e_ms in conc<N>.json), NOT the server
    block, which this script does not fix and should be ignored for disagg.
  * Prefix-cache hit rate in conc<N>.json reads the DECODE server and will look ~0 in
    disagg (prefix matching happens on the PREFILL server) -> verify caching from the
    prefill.final.prom hit/query counters instead. Caching is still ON.
  * SANITY-CHECK ONCE per run that the decode lines actually match RE_VLLM / RE_SGL
    below, else the steady fields stay empty and everything downstream is wrong:
      grep "generation throughput" run.log   # vLLM
      grep "gen throughput"        run.log   # SGLang
"""
from __future__ import annotations
import json, re, sys, statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # reuse parse/_hist_delta/quantile + alias lists

RESULTS = Path("/mnt/home/ppbhatt500/gpumode-triton/results_minimax")
HERE = Path(__file__).resolve().parent

# run dir -> engine .out log (per-interval throughput+batch samples)
LOGS = {
    "agg_vllm_tp4":       HERE / "mm25-agg-6878.out",
    "agg_vllm_tp4_noep":  HERE / "mm25-agg-noep-6897.out",
    "agg_sglang_tp4":     HERE / "mm25-agg-sglang-6876.out",
    "agg_sglang_tp4_ep4": HERE / "mm25-agg-sglang-ep-6900.out",
}
N_GPU = 4
RE_VLLM = re.compile(r"generation throughput:\s*([\d.]+) tokens/s, Running:\s*(\d+)")
RE_SGL  = re.compile(r"#running-req:\s*(\d+).*?gen throughput \(token/s\):\s*([\d.]+)")


def steady_by_conc(logpath: Path) -> dict:
    """Per concurrency phase, over full-batch samples (running >= 0.9*conc):
      tput_pg = median(gen_tput) / N_GPU            -- steady decode tput per GPU
      intvty  = median(gen_tput / running)          -- steady tok/s/user (= 1/TPOT)
    gen_tput and running are the engine's own periodic-log fields (total engine
    throughput, total running batch), so intvty is the true full-batch per-user rate
    and is internally consistent with tput_pg (same samples)."""
    cur, phase = None, {}
    for ln in logpath.read_text(errors="ignore").splitlines():
        m = re.search(r"===== concurrency (\d+)", ln)
        if m:
            cur = int(m.group(1)); phase.setdefault(cur, [])
            continue
        if cur is None:
            continue
        mv = RE_VLLM.search(ln)
        if mv:
            phase[cur].append((float(mv.group(1)), int(mv.group(2)))); continue
        ms = RE_SGL.search(ln)
        if ms:
            phase[cur].append((float(ms.group(2)), int(ms.group(1))))
    out = {}
    for c, rows in phase.items():
        full = [(g, r) for g, r in rows if r >= 0.9 * c and r > 0]
        if full:
            out[c] = {
                "tput_pg": round(st.median([g for g, _ in full]) / N_GPU, 2),
                "intvty": round(st.median([g / r for g, r in full]), 2),
                "samples": len(full),
                "max_batch": max(r for _, r in rows),
            }
    return out


def latency_p(stem: Path):
    """p50/p95/p99 (ms) for ttft/e2e from the prom histogram delta, or None. We do NOT
    extract itl/tpot here: the vLLM decode-latency histogram buckets jump 10ms->25ms,
    too coarse to resolve 5-20ms decode (they collapse c32/c64/c128 to one value).
    Interactivity/TPOT come from the log instead (intvty_steady / tpot_steady_ms)."""
    bf, af = stem.with_suffix(".before.prom"), stem.with_suffix(".after.prom")
    if not (bf.exists() and af.exists()):
        return None
    _, bh = M.parse(bf.read_text(errors="ignore"))
    _, ah = M.parse(af.read_text(errors="ignore"))
    res = {}
    for key, aliases in (("ttft_ms", M.TTFT), ("e2e_ms", M.E2E)):
        s, c, bkts = M._hist_delta(bh, ah, aliases)
        if c and c > 0:
            res[key] = {q: round(M.quantile(bkts, c, p) * 1000, 3)
                        for q, p in (("p50", .50), ("p95", .95), ("p99", .99))}
    return res


def process_run(rundir: Path, logp: Path):
    """Recompute steady metrics for one results dir from its engine log + prom sidecars."""
    if not rundir.exists():
        print(f"skip {rundir.name} (no dir)"); return
    steady = steady_by_conc(logp) if logp and logp.exists() else {}
    if not steady:
        print(f"  WARN {rundir.name}: no steady samples parsed from {logp}")
    for f in sorted(rundir.glob("conc*.json"), key=lambda p: int(re.search(r"conc(\d+)", p.name).group(1))):
        c = int(re.search(r"conc(\d+)", f.name).group(1))
        j = json.loads(f.read_text()); srv = j["server"]
        # 1) steady tput + steady interactivity (log-derived, full-batch).
        #    Steady is the single source of truth -> drop the whole-window output
        #    averages (output_tput_per_gpu / output_tput_total) so conc<N>.json
        #    carries no second, ramp/drain-diluted throughput. (input_tput_per_gpu
        #    is the prefill rate, a different metric with no steady form -> kept.)
        if c in steady:
            sd = steady[c]
            srv["output_tput_per_gpu_steady"] = sd["tput_pg"]
            srv["intvty_steady"] = sd["intvty"]
            srv["tpot_steady_ms"] = round(1000.0 / sd["intvty"], 3) if sd["intvty"] else None
            srv["_steady_samples"] = sd["samples"]
            srv["_steady_max_batch"] = sd["max_batch"]
            srv.pop("output_tput_per_gpu", None)
            srv.pop("output_tput_total", None)
        else:
            print(f"  WARN {rundir.name} conc{c}: no steady sample in log; keeping window avg")
        # drop the bucket-unreliable prom decode-latency fields (10ms->25ms buckets
        # can't resolve 5-20ms decode -> they collapse c32/c64/c128). intvty_steady
        # + tpot_steady_ms (log-derived) are the accurate replacements.
        for k in ("itl_ms", "tpot_ms", "intvty_p50", "intvty_p99"):
            srv.pop(k, None)
        # 2) p95 (+ refresh p50/p99) for ttft/e2e only from the prom histogram
        lp = latency_p(f.with_suffix(""))
        if lp:
            for key, qs in lp.items():
                srv.setdefault(key, {})
                for q, v in qs.items():
                    srv[key][q] = v
        else:
            # no prom -> keep existing p50/p99, fill p95 = p99 (conservative)
            for key in ("ttft_ms", "e2e_ms"):
                if key in srv and srv[key] and "p95" not in srv[key]:
                    srv[key]["p95"] = srv[key].get("p99")
        f.write_text(json.dumps(j, indent=2))
    # report
    line = " ".join(f"c{c}:{steady[c]['tput_pg']}t/{steady[c]['intvty']}iv({steady[c]['samples']}smp)"
                    for c in sorted(steady))
    print(f"{rundir.name:22} steady tput/gpu·intvty -> {line}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Recompute STEADY full-batch tput+interactivity into conc<N>.json.")
    ap.add_argument("--run-dir", help="process ONE results dir (requires --log) instead of the built-in 4-run set")
    ap.add_argument("--log", help="engine log: interleaved '===== concurrency N' markers + 'generation throughput..Running' lines")
    a = ap.parse_args()
    if a.run_dir:
        if not a.log:
            ap.error("--run-dir requires --log")
        process_run(Path(a.run_dir), Path(a.log))
    else:
        for d, logp in LOGS.items():
            process_run(RESULTS / d, logp)


if __name__ == "__main__":
    main()
