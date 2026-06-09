#!/usr/bin/env bash
# Stage 1-3, DISAGG 1P1D (SGLang): serve MiniMax-M2.5 DISAGGREGATED on one node —
# 1 prefill GPU (TP1, --disaggregation-mode prefill, :31000, bootstrap :8998) +
# 1 decode GPU (TP1, --disaggregation-mode decode, :31001). KV moves P->D over the
# nixl transfer backend (UCX rides cuda_ipc/NVLink on-node, no RDMA setup needed).
# The OFFICIAL sglang router (`sglang_router.launch_router --pd-disaggregation`, :8000)
# fronts both as ONE OpenAI endpoint. Then the EXACT SAME sweep as aggregate.
#
# Prefix caching: prefill keeps RadixAttention ON (where the long-prefix agentic reuse
# lives). On the decode server sglang force-disables radix by default; we re-enable it
# with --disaggregation-decode-enable-radix-cache (experimental — remove if unstable).
# sglang auto-disables CUDA graphs on the PREFILL server in disagg mode (its own hook).
# No HiCache here (composability with PD transfer unverified). EP does not apply (TP1).
# Per-GPU numbers keep the harness n_gpu=4 denominator (= per quarter-node; only 2 of
# the 4 GPUs are used).
#
# Submit (same container as aggregate/run_agg_sglang.sbatch):
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 sbatch --job-name=mm25-disagg-sglang-1p1d \
#     --partition=gb300 --nodes=1 --exclusive --gres=gpu:4 --mem=0 --time=12:00:00 --output=%x-%j.out \
#     --wrap "srun --container-image=/mnt/vast/squash_dupe/lmsysorg_sglang_nightly-dev-cu13-20260602-98a1b58c_arm64.sqsh \
#       --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
#       bash /mnt/home/ppbhatt500/gpumode-triton/replay-minimax/disagg/serve_disagg_sglang_1p1d.sh"
#
# Env knobs: MODEL, MAX_MODEL_LEN, CONCURRENCIES, PREARRANGE_FRAC, RAMP_SECONDS,
#   DATASET, RESULT_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"   # replay-minimax/ (sweep.py, metrics.py, dataset)
: "${MODEL:?set MODEL to the MiniMax-M2.5 path}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-196608}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"
PREARRANGE_FRAC="${PREARRANGE_FRAC:-0.75}"
RAMP_SECONDS="${RAMP_SECONDS:-60}"
DATASET="${DATASET:-$ROOT/batch_long.replay.jsonl}"
RESULT_DIR="${RESULT_DIR:-$ROOT/../results_minimax/disagg_sglang_1p1d}"
mkdir -p "$RESULT_DIR"

# run.log = DECODE engine + sweep markers ONLY (what recompute_steady parses). The
# prefill server's '#running-req .. gen throughput' lines (~0 tok/s in disagg) would
# poison the steady medians -> prefill.log; router -> router.log.
RUN_LOG="$RESULT_DIR/run.log"; : > "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

# nixl transfer backend + official router (cu13 nixl wheel matches this image; the
# bare 'nixl' meta-pkg pulls BOTH cuda majors -> ImportError risk).
python3 -c "import nixl" 2>/dev/null || \
  python3 -m pip install -q nixl-cu13 2>/dev/null || python3 -m pip install -q nixl
python3 -c "import sglang_router" 2>/dev/null || python3 -m pip install -q sglang-router

# Same engine flags as serve_agg_sglang.sh (radix/prefix ON, prom at /metrics), plus
# the PD-disagg role flags. No HiCache (see header).
echo "[disagg-sgl-1p1d] prefill GPU0 :31000 (bootstrap :8998, nixl) -> prefill.log"
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name minimax \
  --tp 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "$MAX_MODEL_LEN" \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998 \
  --host 0.0.0.0 --port 31000 \
  > "$RESULT_DIR/prefill.log" 2>&1 &
PRE=$!

echo "[disagg-sgl-1p1d] decode GPU1 :31001 (decode radix re-enabled) -> run.log"
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name minimax \
  --tp 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "$MAX_MODEL_LEN" \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-decode-enable-radix-cache \
  --host 0.0.0.0 --port 31001 &
DEC=$!
trap 'kill $PRE $DEC 2>/dev/null || true' EXIT

echo "[disagg-sgl-1p1d] waiting for /health on :31000 and :31001 ..."
for PORT in 31000 31001; do
  for i in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[disagg-sgl-1p1d] :$PORT healthy"; break; fi
    if ! kill -0 $PRE $DEC 2>/dev/null; then echo "[disagg-sgl-1p1d] a server died during startup" >&2; exit 1; fi
    sleep 10
  done
done

# Official router: one OpenAI endpoint on :8000; bootstrap port rides as the 2nd
# token of --prefill.
echo "[disagg-sgl-1p1d] router :8000 -> P[:31000 bs:8998] D[:31001] (router.log)"
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:31000 8998 \
  --decode  http://127.0.0.1:31001 \
  --host 0.0.0.0 --port 8000 > "$RESULT_DIR/router.log" 2>&1 &
RTR=$!
trap 'kill $PRE $DEC $RTR 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then echo "[disagg-sgl-1p1d] router healthy"; break; fi
  if ! kill -0 $RTR 2>/dev/null; then echo "[disagg-sgl-1p1d] router died (see router.log)" >&2; exit 1; fi
  sleep 2
done

# replayer deps in an isolated venv (matches the colocated launcher pattern)
python3 -m pip install --break-system-packages -q -r "$ROOT/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$ROOT/requirements.txt" || true

# GPU hardware telemetry for the whole sweep (power / util / temp / mem, 1 Hz).
echo "[disagg-sgl-1p1d] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $PRE $DEC $RTR $GPUMON 2>/dev/null || true' EXIT

# Sweep through the ROUTER (:8000); prom ground truth from the DECODE server (:31001).
# Decode-side TTFT excludes prefill compute -> mildly optimistic, documented not hidden.
echo "[disagg-sgl-1p1d] starting sweep -> $RESULT_DIR"
cd "$ROOT"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:8000" \
  --metrics-url "http://localhost:31001/metrics" \
  --model minimax --n-gpu 4 \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 disagg SGLang 1P1D"
# prefill-side counters (radix cache hits live there in disagg) for reference
curl -s "http://localhost:31000/metrics" > "$RESULT_DIR/prefill.final.prom" || true

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true

# Make the metrics correct: recompute_steady swaps the raw whole-window throughput and
# bucket-coarse decode latencies for STEADY full-batch tput + interactivity (log-derived,
# running>=0.9*conc) in each conc<N>.json; then --replot rebuilds pareto.csv/png from the
# corrected JSONs (sweep wrote them mid-run, before recompute, so their steady cols are empty).
sync; sleep 3
echo "[disagg-sgl-1p1d] recompute_steady -> steady full-batch tput/interactivity from $RUN_LOG"
python3 recompute_steady.py --run-dir "$RESULT_DIR" --log "$RUN_LOG" \
  || echo "[disagg-sgl-1p1d] recompute_steady FAILED — rerun: python3 recompute_steady.py --run-dir '$RESULT_DIR' --log '$RUN_LOG'"
echo "[disagg-sgl-1p1d] replot pareto from corrected conc<N>.json"
python3 sweep.py --replot --dataset "$DATASET" --result-dir "$RESULT_DIR" \
  --title "MiniMax-M2.5 disagg SGLang 1P1D" \
  || echo "[disagg-sgl-1p1d] replot FAILED — rerun: python3 sweep.py --replot --dataset '$DATASET' --result-dir '$RESULT_DIR'"
echo "[disagg-sgl-1p1d] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
