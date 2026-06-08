#!/usr/bin/env python3
"""INDEPENDENT prom-truth auditor for the InferenceX dump.

Re-derives every plotted latency/throughput number straight from the raw
Prometheus exposition (NOT by calling make_inferencex_dump_prom's helpers) and
checks it against ~/InferenceX-app/local-dump/benchmark_results.json.

Disagg invariants proven here (the risky path):
  I1 MONOTONIC   every cumulative *_count / *_sum / *_bucket used is
                 non-decreasing across the conc window  -> window-diff is valid.
  I2 WINDOW      histogram count-delta over [mtime-runtime, mtime] == the run's
                 completed_turns (+/- small)            -> window isolates the run.
  I3 QUANTILE    independently recomputed mean/p50/p90/p99 == dump (exact).
  I4 THROUGHPUT  prom output-token delta == conc.json output_tokens_total
                 (the count the dump's throughput is built from)  -> client
                 token count is prom-equivalent; on a counter that RESETS we
                 fall back to summing positive increments and say so.
Colocated configs are re-derived from c<N>_prom.json 'delta' the same way.
"""
import datetime as dt
import json
import os
import re
from pathlib import Path

RESULTS = Path("/mnt/home/ppbhatt500/gpumode-triton/results")
DUMP = Path.home() / "InferenceX-app" / "local-dump" / "benchmark_results.json"
CONF = Path.home() / "InferenceX-app" / "local-dump" / "configs.json"

FAM = {
    "ttft": ("vllm:time_to_first_token_seconds", "sglang:time_to_first_token_seconds", "dynamo_frontend_time_to_first_token_seconds"),
    "tpot": ("vllm:inter_token_latency_seconds", "sglang:inter_token_latency_seconds", "dynamo_frontend_inter_token_latency_seconds"),
    "e2e":  ("vllm:e2e_request_latency_seconds", "sglang:e2e_request_latency_seconds", "dynamo_frontend_request_duration_seconds"),
}
OUT_TOK = ("vllm:generation_tokens_total", "sglang:generation_tokens_total", "dynamo_frontend_output_tokens_total")

RUNS = [
    ("dsv4_b1",          "vllm",   0, False, [4, 8, 16, 32, 64, 128], 4),
    ("dsv4_b1_sglang",   "sglang", 1, False, [4, 8, 16, 32, 64, 128], 4),
    ("kimi_b1",          "vllm",   0, False, [4, 8, 16, 32, 64, 128], 4),
    ("kimi_b1_sglang",   "sglang", 1, False, [4, 8, 16, 32, 64, 128], 4),
    ("dsv4_vllm_4p_4d",   "dynamo", 2, True,  [4, 8, 16], 8),
    ("dsv4_vllm_8p_4d",   "dynamo", 2, True,  [4, 8, 16], 12),
    ("dsv4_vllm_4p_8d",   "dynamo", 2, True,  [4, 8, 16], 12),
    ("dsv4_sglang_4p_4d", "dynamo", 2, True,  [4, 8, 16], 8),
    ("dsv4_sglang_4p_8d", "dynamo", 2, True,  [4, 8, 16], 12),
]


def q(buckets, total, p):
    if total <= 0 or not buckets:
        return 0.0
    rank = p * total
    prev_le, prev_cum = 0.0, 0.0
    for le, cum in buckets:
        if cum >= rank:
            if le == float("inf"):
                return prev_le
            if cum == prev_cum:
                return le
            return prev_le + (rank - prev_cum) / (cum - prev_cum) * (le - prev_le)
        if le != float("inf"):
            prev_le = le
        prev_cum = cum
    return prev_le


def stats(s, c, buckets):
    mean = (s / c * 1000) if c else 0.0
    return (round(mean, 4), round(q(buckets, c, .5) * 1000, 4),
            round(q(buckets, c, .9) * 1000, 4), round(q(buckets, c, .99) * 1000, 4))


# ---- colocated ----
def colo_hist(delta, prefix):
    s = cnt = None
    bk = []
    for k, v in delta.items():
        if k.startswith(prefix + "_sum"):
            s = v
        elif k.startswith(prefix + "_count"):
            cnt = v
        elif k.startswith(prefix + "_bucket"):
            m = re.search(r'le="([^"]+)"', k)
            if m:
                le = float("inf") if m.group(1) in ("+Inf", "Inf") else float(m.group(1))
                bk.append((le, v))
    bk.sort()
    if cnt is not None and (not bk or bk[-1][0] != float("inf")):
        bk.append((float("inf"), cnt))
    return s, cnt, bk


# ---- disagg: parse ALL snapshots, keep full series for monotonicity ----
def parse_all(path, name):
    """Return list of (ts, dict) where dict has name_sum/_count/_bucket|le and OUT total."""
    snaps = []
    ts = None
    cur = {}
    want = [FAM[f][2] for f in FAM] + [OUT_TOK[2]]
    with open(path) as fh:
        for line in fh:
            if line.startswith("# ==== ts="):
                if ts is not None:
                    snaps.append((ts, cur))
                ts = dt.datetime.fromisoformat(line.split("ts=")[1].split(" ")[0])
                cur = {}
                continue
            if not line.startswith("dynamo_frontend_"):
                continue
            if not any(w in line for w in want):
                continue
            m = re.match(r"(\S+?)(\{[^}]*\})?\s+([0-9eE+.\-]+)\s*$", line.strip())
            if not m:
                continue
            nm, labels, val = m.group(1), m.group(2) or "", m.group(3)
            lm = re.search(r'le="([^"]+)"', labels)
            key = nm + ("|" + lm.group(1) if lm else "")
            cur[key] = float(val)
    if ts is not None:
        snaps.append((ts, cur))
    return snaps


def snap_at(snaps, t):
    best = snaps[0][1]
    for ts, c in snaps:
        if ts <= t:
            best = c
        else:
            break
    return best


def in_window(snaps, st, mt):
    return [(ts, c) for ts, c in snaps if st <= ts <= mt]


def hist_diff(a, b, name):
    s = b.get(name + "_sum", 0) - a.get(name + "_sum", 0)
    c = b.get(name + "_count", 0) - a.get(name + "_count", 0)
    les = sorted({k.split("|")[1] for k in b if k.startswith(name + "_bucket|")},
                 key=lambda x: float("inf") if x in ("+Inf", "Inf") else float(x))
    bk = []
    for le in les:
        cum = b.get(f"{name}_bucket|{le}", 0) - a.get(f"{name}_bucket|{le}", 0)
        lef = float("inf") if le in ("+Inf", "Inf") else float(le)
        bk.append((lef, cum))
    if bk and bk[-1][0] != float("inf"):
        bk.append((float("inf"), c))
    return s, c, bk


def monotonic(snaps_win, name):
    """Check name_count and all name_bucket|le are non-decreasing across window."""
    keys = [name + "_count"] + sorted({k for _, c in snaps_win for k in c
                                       if k.startswith(name + "_bucket|")})
    resets = 0
    for k in keys:
        prev = None
        for _, c in snaps_win:
            v = c.get(k)
            if v is None:
                continue
            if prev is not None and v < prev - 1e-9:
                resets += 1
            prev = v
    return resets


def fmt(label, ok):
    return f"  [{'PASS' if ok else 'FAIL'}] {label}"


def main():
    dump = json.loads(DUMP.read_text())
    confs = json.loads(CONF.read_text())
    cid2fw = {c["id"]: (c["framework"], c["model"]) for c in confs}
    # index dump rows by (dir-equivalent) via config framework+model+disagg+gpu; easier: by config_id+conc
    by_key = {}
    for r in dump:
        by_key[(r["config_id"], r["conc"])] = r["metrics"]
    cid_of = {}
    # rebuild config_id order == RUNS order (generator enumerates RUNS start=1)
    for i, run in enumerate(RUNS, start=1):
        cid_of[run[0]] = i

    all_ok = True
    for d, eng, bi, disagg, concs, total_gpu in RUNS:
        cid = cid_of[d]
        print(f"\n=== {d}  ({'disagg' if disagg else 'colocated'}, eng={eng}, gpus={total_gpu}) ===")
        snaps = parse_all(RESULTS / d / "server_metrics.prom", d) if eng == "dynamo" else None
        for c in concs:
            cj_path = RESULTS / d / f"conc{c}.json"
            if not cj_path.exists():
                continue
            cj = json.loads(cj_path.read_text())
            rt = cj["runtime_s"]
            dump_m = by_key.get((cid, c))
            if dump_m is None:
                print(fmt(f"conc{c}: NO DUMP ROW", False)); all_ok = False; continue

            if eng != "dynamo":  # colocated
                delta = json.loads((RESULTS / d / f"c{c}_prom.json").read_text())["delta"]
                H = {f: colo_hist(delta, FAM[f][bi]) for f in FAM}
                out_tok = 0.0
                for k, v in delta.items():
                    if k.startswith(OUT_TOK[bi]):
                        out_tok += v
                turns = H["ttft"][1]
                resets = 0
                win_n = "(delta block)"
            else:  # disagg
                mt = dt.datetime.fromtimestamp(os.path.getmtime(cj_path), tz=dt.timezone.utc)
                st = mt - dt.timedelta(seconds=rt)
                a, b = snap_at(snaps, st), snap_at(snaps, mt)
                win = in_window(snaps, st, mt)
                win_n = f"{len(win)} snaps"
                H = {f: hist_diff(a, b, FAM[f][bi]) for f in FAM}
                resets = sum(monotonic(win, FAM[f][2]) for f in FAM)
                # prom output-token delta (counter may reset)
                otk = OUT_TOK[2]
                end_v, start_v = b.get(otk, 0), a.get(otk, 0)
                prom_out_naive = end_v - start_v
                # positive-increment sum across window (robust to resets)
                prom_out_pos = 0.0
                prev = None
                for _, cc in win:
                    v = cc.get(otk)
                    if v is None:
                        continue
                    if prev is not None and v >= prev:
                        prom_out_pos += v - prev
                    prev = v
                turns = H["ttft"][1]
                out_tok = cj["output_tokens_total"]

            # I3: independent recompute vs dump (exact, rounded to dump's 6dp)
            recomputed = {}
            for fam, key in (("ttft", "ttft"), ("tpot", "tpot"), ("e2e", "e2el")):
                mean, p50, p90, p99 = stats(*H[fam])
                recomputed[f"mean_{key}"] = round(mean / 1000, 6)
                recomputed[f"median_{key}"] = round(p50 / 1000, 6)
                recomputed[f"p90_{key}"] = round(p90 / 1000, 6)
                recomputed[f"p99_{key}"] = round(p99 / 1000, 6)
            # intvty from tpot
            mt_, p50t, p90t, p99t = stats(*H["tpot"])
            for nm, val in (("mean", mt_), ("median", p50t), ("p90", p90t), ("p99", p99t)):
                recomputed[f"{nm}_intvty"] = round(1000.0 / val, 4) if val else 0.0
            tput = round(out_tok / rt / total_gpu, 4)
            recomputed["tput_per_gpu"] = tput

            q3_bad = [k for k, v in recomputed.items()
                      if abs((dump_m.get(k) or 0) - v) > 5e-6]
            i3 = not q3_bad

            # I2: count delta vs completed_turns
            exp_turns = cj["completed_turns"]
            i2 = abs(turns - exp_turns) <= max(2, 0.01 * exp_turns)

            # I1: monotonic (disagg only)
            i1 = (resets == 0)

            # I4: prom out-token == conc.json count (throughput source)
            if eng == "dynamo":
                drift_naive = abs(prom_out_naive - out_tok) / out_tok if out_tok else 1
                drift_pos = abs(prom_out_pos - out_tok) / out_tok if out_tok else 1
                i4 = drift_naive <= 0.02 or drift_pos <= 0.02
                src = ("naive-diff" if drift_naive <= 0.02 else
                       f"pos-incr (naive reset {prom_out_naive:.0f}!={out_tok})")
                i4d = f"prom_out={prom_out_pos:.0f} cj={out_tok} drift_pos={drift_pos*100:.2f}% [{src}]"
            else:
                drift = abs(out_tok - cj["output_tokens_total"]) / cj["output_tokens_total"]
                i4 = drift <= 0.02
                i4d = f"prom_out_delta={out_tok:.0f} cj={cj['output_tokens_total']} drift={drift*100:.2f}%"

            ok = i1 and i2 and i3 and i4
            all_ok = all_ok and ok
            tag = "PASS" if ok else "FAIL"
            print(f"  conc{c:<4} {win_n:<12} count={turns:.0f}/{exp_turns} "
                  f"[I1mono={'ok' if i1 else f'{resets} RESETS'}] "
                  f"[I2win={'ok' if i2 else 'OFF'}] [I3quant={'ok' if i3 else 'MISMATCH'}] "
                  f"[I4tput={'ok' if i4 else 'OFF'}] -> {tag}")
            print(f"        I4: {i4d}")
            print(f"        p99: ttft={dump_m['p99_ttft']:.4f}s tpot={dump_m['p99_tpot']*1000:.2f}ms "
                  f"e2e={dump_m['p99_e2el']:.2f}s intvty={dump_m['p99_intvty']:.1f}  "
                  f"(median intvty={dump_m['median_intvty']:.1f})")
            if not i3:
                print(f"        QUANT MISMATCH: {[(k, dump_m.get(k), recomputed[k]) for k in q3_bad]}")

    print("\n" + "=" * 70)
    print("OVERALL:", "ALL PASS — dump == raw prom, disagg windows valid" if all_ok
          else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
