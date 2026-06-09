#!/usr/bin/env bash
# Stage 1-3, DISAGG 1P3D (SGLang): serve MiniMax-M2.5 DISAGGREGATED on one node —
# 1 prefill GPU (TP1, :31000, bootstrap :8998) + 3 DECODE WORKERS (TP1 each, GPUs
# 1/2/3 on :31001/:31002/:31003), all behind the OFFICIAL sglang router (:8000, one
# --decode per worker; its default cache-aware policy gives decode-side session
# affinity). KV moves P->D over the nixl transfer backend (UCX rides cuda_ipc/NVLink
# on-node). Then the EXACT SAME sweep as aggregate.
#
# Why 3x TP1 and not one TP3 decode instance: MiniMax-M2.5 has 8 KV heads, 8 % 3 != 0
# -> TP3 cannot shard attention. EP does not apply (TP1 per instance, like 1P1D).
# Prefix caching: the long-prefix reuse lives on the SINGLE prefill instance
# (RadixAttention ON); decode radix re-enabled on each worker (experimental flag).
# CAVEATS: sweep scrapes decode #1 only (~1/3 of traffic; counters undercount 3x);
# recompute_steady's full-batch filter (running >= 0.9*conc) never fires with 3
# interleaved decode engines -> it WARNs and conc<N>.json keeps the whole-window
# prom numbers. Documented, not hidden.
#
# Submit (same container as aggregate/run_agg_sglang.sbatch):
#   MODEL=/mnt/vast/models/minimax-m2.5-nvfp4 sbatch --job-name=mm25-disagg-sglang-1p3d \
#     --partition=gb300 --nodes=1 --exclusive --gres=gpu:4 --mem=0 --time=12:00:00 --output=%x-%j.out \
#     --wrap "srun --container-image=/mnt/vast/squash_dupe/lmsysorg_sglang_nightly-dev-cu13-20260602-98a1b58c_arm64.sqsh \
#       --container-mounts=/mnt/home:/mnt/home,/mnt/vast:/mnt/vast \
#       bash /mnt/home/ppbhatt500/gpumode-triton/replay-minimax/disagg/serve_disagg_sglang_1p3d.sh"
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
RESULT_DIR="${RESULT_DIR:-$ROOT/../results_minimax/disagg_sglang_1p3d}"
mkdir -p "$RESULT_DIR"

# run.log = the 3 DECODE engines + sweep markers (see caveats above). The prefill
# server's '#running-req .. gen throughput' lines would poison it worse -> prefill.log.
RUN_LOG="$RESULT_DIR/run.log"; : > "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

# nixl transfer backend + official router (cu13 nixl wheel matches this image)
python3 -c "import nixl" 2>/dev/null || \
  python3 -m pip install -q nixl-cu13 2>/dev/null || python3 -m pip install -q nixl
python3 -c "import sglang_router" 2>/dev/null || python3 -m pip install -q sglang-router

# Same engine flags as serve_agg_sglang.sh, plus the PD-disagg role flags. No HiCache.
echo "[disagg-sgl-1p3d] prefill GPU0 :31000 (bootstrap :8998, nixl) -> prefill.log"
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

echo "[disagg-sgl-1p3d] decode1 GPU1 :31001 (decode radix re-enabled) -> run.log"
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
DEC1=$!

echo "[disagg-sgl-1p3d] decode2 GPU2 :31002 (decode radix re-enabled) -> run.log"
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server \
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
  --host 0.0.0.0 --port 31002 &
DEC2=$!

echo "[disagg-sgl-1p3d] decode3 GPU3 :31003 (decode radix re-enabled) -> run.log"
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server \
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
  --host 0.0.0.0 --port 31003 &
DEC3=$!
trap 'kill $PRE $DEC1 $DEC2 $DEC3 2>/dev/null || true' EXIT

echo "[disagg-sgl-1p3d] waiting for /health on :31000 :31001 :31002 :31003 ..."
for PORT in 31000 31001 31002 31003; do
  for i in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[disagg-sgl-1p3d] :$PORT healthy"; break; fi
    if ! kill -0 $PRE $DEC1 $DEC2 $DEC3 2>/dev/null; then echo "[disagg-sgl-1p3d] a server died during startup" >&2; exit 1; fi
    sleep 10
  done
done

# Official router: one OpenAI endpoint on :8000; bootstrap port rides as the 2nd
# token of --prefill; one --decode per worker.
echo "[disagg-sgl-1p3d] router :8000 -> P[:31000 bs:8998] D[:31001 :31002 :31003] (router.log)"
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:31000 8998 \
  --decode  http://127.0.0.1:31001 \
  --decode  http://127.0.0.1:31002 \
  --decode  http://127.0.0.1:31003 \
  --host 0.0.0.0 --port 8000 > "$RESULT_DIR/router.log" 2>&1 &
RTR=$!
trap 'kill $PRE $DEC1 $DEC2 $DEC3 $RTR 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then echo "[disagg-sgl-1p3d] router healthy"; break; fi
  if ! kill -0 $RTR 2>/dev/null; then echo "[disagg-sgl-1p3d] router died (see router.log)" >&2; exit 1; fi
  sleep 2
done

# replayer deps in an isolated venv (matches the colocated launcher pattern)
python3 -m pip install --break-system-packages -q -r "$ROOT/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$ROOT/requirements.txt" || true

# GPU hardware telemetry for the whole sweep (power / util / temp / mem, 1 Hz).
echo "[disagg-sgl-1p3d] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $PRE $DEC1 $DEC2 $DEC3 $RTR $GPUMON 2>/dev/null || true' EXIT

# Sweep through the ROUTER (:8000); prom from decode #1 (:31001, ~1/3 of traffic —
# see header caveats). All decoders + prefill snapshot their final /metrics afterwards.
echo "[disagg-sgl-1p3d] starting sweep -> $RESULT_DIR"
cd "$ROOT"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:8000" \
  --metrics-url "http://localhost:31001/metrics" \
  --model minimax --n-gpu 4 \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 disagg SGLang 1P3D"
curl -s "http://localhost:31000/metrics" > "$RESULT_DIR/prefill.final.prom" || true
curl -s "http://localhost:31001/metrics" > "$RESULT_DIR/decode1.final.prom" || true
curl -s "http://localhost:31002/metrics" > "$RESULT_DIR/decode2.final.prom" || true
curl -s "http://localhost:31003/metrics" > "$RESULT_DIR/decode3.final.prom" || true

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true

# recompute_steady will WARN here (3 interleaved decode engines never satisfy the
# single-engine full-batch filter) and keep the window averages — expected, see header.
sync; sleep 3
echo "[disagg-sgl-1p3d] recompute_steady (expect 'no steady sample' WARNs — see header)"
python3 recompute_steady.py --run-dir "$RESULT_DIR" --log "$RUN_LOG" \
  || echo "[disagg-sgl-1p3d] recompute_steady FAILED — rerun: python3 recompute_steady.py --run-dir '$RESULT_DIR' --log '$RUN_LOG'"
echo "[disagg-sgl-1p3d] replot pareto from conc<N>.json"
python3 sweep.py --replot --dataset "$DATASET" --result-dir "$RESULT_DIR" \
  --title "MiniMax-M2.5 disagg SGLang 1P3D" \
  || echo "[disagg-sgl-1p3d] replot FAILED — rerun: python3 sweep.py --replot --dataset '$DATASET' --result-dir '$RESULT_DIR'"
echo "[disagg-sgl-1p3d] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
