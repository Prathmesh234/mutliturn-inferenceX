#!/usr/bin/env bash
# DIAGNOSTIC variant of serve_agg.sh: same MiniMax-M2.5 aggregated vLLM TP+EP server,
# but with a MUCH larger prefill/decode token budget to test whether the ~800 tok/s/gpu
# ceiling is a serving-config cap (max-num-batched-tokens) rather than a roofline limit.
# Differs from serve_agg.sh ONLY in: --max-num-batched-tokens (32768 vs 8192),
# an explicit --max-num-seqs, and a dedicated RESULT_DIR. EP stays ON (TP+EP).
#
# Env knobs (same as serve_agg.sh) plus:
#   MAX_NUM_BATCHED  prefill+decode token budget per step   (default 32768)
#   MAX_NUM_SEQS     max concurrent running requests        (default 256)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL:?set MODEL to the MiniMax-M2.5 path}"
TP="${TP:-4}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"; OFFLOAD_GB="${OFFLOAD_GB:-0}"
CONCURRENCIES="${CONCURRENCIES:-128}"; PREARRANGE_FRAC="${PREARRANGE_FRAC:-0.75}"
RAMP_SECONDS="${RAMP_SECONDS:-60}"; DATASET="${DATASET:-$HERE/batch_long.replay.jsonl}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-32768}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
PORT="${PORT:-8000}"
ENGINE_TAG="vllm_tp${TP}_ep_bb"          # bb = big-batch diagnostic
RESULT_DIR="${RESULT_DIR:-$HERE/../results_minimax/agg_${ENGINE_TAG}}"
mkdir -p "$RESULT_DIR"

echo "[serve-bb] vllm serve $MODEL  TP=$TP EP=on  max-num-batched-tokens=$MAX_NUM_BATCHED  max-num-seqs=$MAX_NUM_SEQS"
vllm serve "$MODEL" \
  --served-model-name minimax \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --port "$PORT" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

echo "[serve-bb] waiting for /health on :$PORT ..."
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[serve-bb] healthy"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[serve-bb] server died during startup" >&2; exit 1; fi
  sleep 10
done

python3 -m pip install --break-system-packages -q -r "$HERE/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$HERE/requirements.txt" || true

echo "[serve-bb] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $SRV $GPUMON 2>/dev/null || true' EXIT

echo "[serve-bb] starting sweep ($CONCURRENCIES) -> $RESULT_DIR"
cd "$HERE"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:$PORT" \
  --metrics-url "http://localhost:$PORT/metrics" \
  --model minimax --n-gpu "$TP" \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 agg TP$TP EP big-batch (mnbt=$MAX_NUM_BATCHED)"

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true
echo "[serve-bb] done -> $RESULT_DIR"
