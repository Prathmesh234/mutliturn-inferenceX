#!/usr/bin/env python3
"""vLLM-EP (MiniMax-M2.5 NVFP4) throughput-vs-(interactivity/TTFT/E2E) pareto, on
STEADY-STATE ground truth, with the first-principles decode roofline overlaid.

Provenance (all engine-side, no client timers):
  - throughput T/gpu  = STEADY: median engine-log `generation throughput` over
    full-batch samples (running >= 0.9*conc) / 4 GPUs  (recompute_steady.py). The
    whole-window average was removed — it was diluted up to ~2.2x by ramp/drain.
  - interactivity I   = STEADY: median(gen_tput / running-batch) over the SAME
    samples = true full-batch tok/s/user (= 1/TPOT). Consistent with T (same samples).
  - TTFT / E2E (p99)  = server /metrics histogram over the window (no steady form —
    engines don't log per-request TTFT).
  - effective batch B = T_total / I  (self-consistent: gen_tput = B * I).

Decode roofline:  per-user step time t(B) = t_fix + t_var*B  (HBM-bandwidth bound).
  I(B) = 1/t(B);  T/gpu(B) = B/(N_GPU*t(B)) -> ceiling 1/(N_GPU*t_var) as B grows.
Fit t_fix,t_var by least squares on the steady (B, 1/I) points. With steady data the
engine SUSTAINS the requested batch at every point, so throughput rises along the
roofline toward the ceiling — there is NO saturation plateau (the earlier plateau was
the ramp/drain dilution of the whole-window average, now removed). Interactivity falls
as ~1/t(B): the normal batch-vs-latency trade, not a collapse.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_GPU = 4
CONCS = [8, 16, 32, 64, 128]          # c4 dropped (latency-pinned outlier)
D = "agg_vllm_tp4"

I, T, ttft, e2e, B = [], [], [], [], []
for c in CONCS:
    s = json.load(open(f"{D}/conc{c}.json"))["server"]
    tpg = s["output_tput_per_gpu_steady"]      # steady tput/gpu (log-derived)
    iv = s["intvty_steady"]                     # steady tok/s/user (log-derived)
    I.append(iv)
    T.append(tpg)
    ttft.append(s["ttft_ms"]["p99"])
    e2e.append(s["e2e_ms"]["p99"] / 1000.0)
    B.append(tpg * N_GPU / iv)                  # effective steady batch = gen_tput / I
I = np.array(I); T = np.array(T); B = np.array(B)

# --- fit decode roofline t(B) = t_fix + t_var*B on steady (B, 1/I), seconds ---
t = 1.0 / I                                     # s per output token per user
t_fix, t_var = np.linalg.lstsq(np.vstack([np.ones_like(B), B]).T, t, rcond=None)[0]
T_ceiling = 1.0 / (N_GPU * t_var)               # HBM-bound tput/gpu ceiling (B -> inf)
frac = T.max() / T_ceiling
print(f"roofline fit: t(B)={t_fix*1e3:.2f}ms + {t_var*1e3:.3f}ms*B  ->  "
      f"ceiling {T_ceiling:.0f} tok/s/gpu (c128 at {frac*100:.0f}% of it)")

# model curves over a batch grid extended past the measured range toward the ceiling
Bg = np.linspace(B.min(), B.max() * 2.2, 240)
tg = t_fix + t_var * Bg
I_model = 1.0 / tg
T_model = Bg / (N_GPU * tg)

BLUE, GREEN, RED = "#1f4e8c", "#2e8b57", "#c0392b"
fig, ax = plt.subplots(1, 3, figsize=(19, 7.4))
fig.subplots_adjust(top=0.66, bottom=0.10, wspace=0.27, left=0.05, right=0.985)


def scatter(a, x, label):
    for xx, yy, c, b in zip(x, T, CONCS, B):
        a.plot(xx, yy, "o", ms=10, color=BLUE, zorder=5)
        a.annotate(f"c{c}\nB≈{b:.0f}", (xx, yy), textcoords="offset points",
                   xytext=(8, -2), fontsize=8.5, color=BLUE)
    a.set_xlabel(label); a.set_ylabel("Output throughput / GPU  (tok/s)")
    a.grid(True, alpha=0.3)


# Panel 1: throughput vs interactivity — steady empirical points + decode roofline
order = np.argsort(CONCS)
ax[0].plot(I[order], T[order], "-", color="#888", lw=1.3, zorder=1)   # empirical path
ax[0].plot(I_model, T_model, "-", color=GREEN, lw=1.9, zorder=2,
           label=f"decode roofline  t(B)={t_fix*1e3:.1f}+{t_var*1e3:.2f}·B ms")
scatter(ax[0], I, "Interactivity  (tok/s/user, steady = 1/TPOT)")
ax[0].axhline(T_ceiling, color=RED, ls=":", lw=1.7,
              label=f"HBM-bound ceiling ≈{T_ceiling:.0f} tok/s/gpu  (B→∞)")
ax[0].annotate(f"steady: batch ↑ ⇒ throughput ↑ toward ceiling,\n"
               f"interactivity ↓ as 1/t(B) — no plateau\n(c128 = {frac*100:.0f}% of ceiling)",
               (I.min() * 1.02, T_ceiling * 0.5), fontsize=8.5, color=RED, ha="left")
ax[0].set_title("Throughput vs Interactivity\n(steady pareto + decode roofline)", fontsize=11)
ax[0].legend(loc="upper right", fontsize=8.5, framealpha=0.92)

scatter(ax[1], ttft, "TTFT p99  (ms)")
ax[1].set_title("Throughput vs TTFT p99\n(prefill / queueing)", fontsize=11)
scatter(ax[2], e2e, "E2E p99  (s)")
ax[2].set_title("Throughput vs E2E p99\n(TTFT + len·TPOT + queueing)", fontsize=11)

cfg = ("vLLM EP  —  MiniMax-M2.5 (NVFP4 / ModelOpt fp4)  •  vLLM v0.21.0  •  aggregated (colocated prefill+decode), 1 node / 4× GB300\n"
       "TP=4,  EP=4 (--enable-expert-parallel)  •  KV-cache fp8  •  prefix-caching ON (hit ≈99.9%)  •  max-model-len 40960  •  max-num-batched-tokens 8192  •  gpu-mem-util 0.90\n"
       "workload: agentic replay, 150 sessions / 1745 turns, pre-arrange 75%, ramp 60s  •  sweep 8→128  •  T & I = STEADY full-batch (engine-log); TTFT/E2E = prom /metrics; B = T_total/I")
fig.suptitle("MiniMax-M2.5  ·  vLLM Expert-Parallel  ·  STEADY-state pareto frontier vs first-principles decode roofline",
             fontsize=14, y=0.985, fontweight="bold")
fig.text(0.5, 0.91, cfg, ha="center", va="top", fontsize=9.0, family="monospace",
         bbox=dict(boxstyle="round,pad=0.6", fc="#f4f6fa", ec="#9bb0c9"))

out = f"{D}/vllm_ep_firstprinciples.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
