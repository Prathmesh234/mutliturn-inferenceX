#!/usr/bin/env python3
"""Cross-check: every plotted metric in the InferenceX API must trace back to
our source conc*.json via the documented transform. GPUS=4 (colocated TP4)."""
import json
import subprocess
import sys

GPUS = 4
RESULTS = "/mnt/home/ppbhatt500/gpumode-triton/results"
DATE = "2026-06-08"

# (result_dir, db_model, display_model, framework)
SOURCES = [
    ("dsv4_b1",         "dsv4",     "DeepSeek-V4-Pro", "vllm"),
    ("dsv4_b1_sglang",  "dsv4",     "DeepSeek-V4-Pro", "sglang"),
    ("kimi_b1",         "kimik2.5", "Kimi-K2.5",       "vllm"),
    ("kimi_b1_sglang",  "kimik2.5", "Kimi-K2.5",       "sglang"),
]
CONCS = [4, 8, 16]


def load_src(d, c):
    with open(f"{RESULTS}/{d}/conc{c}.json") as f:
        return json.load(f)


def fetch_api(display):
    out = subprocess.check_output(
        ["curl", "-s", "--compressed",
         f"http://localhost:3000/api/v1/benchmarks?model={display}&date={DATE}"]
    )
    return json.loads(out)


def approx(a, b, tol=0.02):
    """Match within 2% relative or 0.01 absolute (rounding in transform)."""
    if a is None or b is None:
        return False
    if abs(a - b) <= 0.01:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


# Pull API data once per display model
api_cache = {}
for _, _, display, _ in SOURCES:
    if display not in api_cache:
        api_cache[display] = fetch_api(display)

rows = []
all_pass = True
for d, dbmodel, display, fw in SOURCES:
    api_rows = api_cache[display]
    for c in CONCS:
        src = load_src(d, c)
        # find matching API row
        match = [r for r in api_rows
                 if r["framework"] == fw and r["conc"] == c and r["model"] == dbmodel]
        if len(match) != 1:
            rows.append((d, c, "ROW", f"expected 1 API row, got {len(match)}", "FAIL"))
            all_pass = False
            continue
        m = match[0]["metrics"]

        expected = {
            "tput_per_gpu":  src["output_throughput_tok_per_s"] / GPUS,
            "input_tput_per_gpu": (src["total_token_throughput_tok_per_s"]
                                   - src["output_throughput_tok_per_s"]) / GPUS,
            "median_ttft":   src["ttft_ms"]["p50"],
            "mean_ttft":     src["ttft_ms"]["mean"],
            "p99_ttft":      src["ttft_ms"]["p99"],
            "median_tpot":   src["tpot_ms"]["p50"],
            "mean_tpot":     src["tpot_ms"]["mean"],
            "p99_tpot":      src["tpot_ms"]["p99"],
            "median_e2el":   src["e2e_ms"]["p50"],
            "mean_e2el":     src["e2e_ms"]["mean"],
            "p99_e2el":      src["e2e_ms"]["p99"],
            "median_intvty": 1000.0 / src["tpot_ms"]["p50"],
        }
        point_ok = True
        bad = []
        for k, exp in expected.items():
            got = m.get(k)
            if not approx(got, exp):
                point_ok = False
                bad.append(f"{k}: api={got} exp={exp:.4f}")
        status = "PASS" if point_ok else "FAIL"
        if not point_ok:
            all_pass = False
        rows.append((f"{display}/{fw}", c, "metrics",
                     "ok" if point_ok else "; ".join(bad), status))

# also assert ISL/OSL nominal per model
NOMINAL = {"dsv4": (5220, 671), "kimik2.5": (5760, 768)}
for _, dbmodel, display, fw in SOURCES:
    for r in api_cache[display]:
        if r["model"] == dbmodel and r["framework"] == fw:
            ni, no = NOMINAL[dbmodel]
            ok = r["isl"] == ni and r["osl"] == no
            if not ok:
                all_pass = False
                rows.append((f"{display}/{fw}", r["conc"], "isl/osl",
                             f"api=({r['isl']},{r['osl']}) exp=({ni},{no})", "FAIL"))

print(f"{'config':<26}{'conc':<6}{'field':<9}{'status':<6}detail")
print("-" * 90)
for cfg, c, field, detail, status in rows:
    print(f"{cfg:<26}{str(c):<6}{field:<9}{status:<6}{detail if status=='FAIL' else ''}")

npass = sum(1 for r in rows if r[4] == "PASS")
nfail = sum(1 for r in rows if r[4] == "FAIL")
print("-" * 90)
print(f"12 expected metric points. PASS={npass} FAIL={nfail}")
print("OVERALL:", "ALL PASS" if all_pass else "FAILURES PRESENT")
sys.exit(0 if all_pass else 1)
