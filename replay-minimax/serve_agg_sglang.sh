#!/usr/bin/env bash
# Stage 1-3, SGLANG variant: serve MiniMax-M2.5 AGGREGATED (colocated) on one node,
# then run the EXACT SAME sweep (same 150 sessions, same pre-arrange 75%, same ramp,
# same prom + GPU metrics). The ONLY difference from serve_agg.sh is the server
# process: SGLang instead of vLLM. Everything downstream (replay_bench.py, sweep.py,
# metrics.py, gpu_metrics.py) is engine-agnostic and shared.
#
# Prefix caching = SGLang's RadixAttention cache, ON by default — we deliberately do
# NOT pass --disable-radix-cache (the dsv4 disagg recipes disabled it, which was wrong
# for agentic; see the dsv4 RCA). --enable-metrics exposes Prometheus at /metrics.
#
# Env knobs (identical to serve_agg.sh): MODEL, TP, MAX_MODEL_LEN, OFFLOAD_GB,
#   CONCURRENCIES, PREARRANGE_FRAC, RAMP_SECONDS, DATASET, RESULT_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL:?set MODEL to the MiniMax-M2.5 path}"
TP="${TP:-4}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"; OFFLOAD_GB="${OFFLOAD_GB:-0}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"; PREARRANGE_FRAC="${PREARRANGE_FRAC:-0.75}"
RAMP_SECONDS="${RAMP_SECONDS:-60}"; DATASET="${DATASET:-$HERE/batch_long.replay.jsonl}"
PORT="${PORT:-8000}"
ENGINE_TAG="sglang_tp${TP}"; { [ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && ENGINE_TAG="${ENGINE_TAG}_offload${OFFLOAD_GB}"; } || true
RESULT_DIR="${RESULT_DIR:-$HERE/../results_minimax/agg_${ENGINE_TAG}}"
mkdir -p "$RESULT_DIR"

OFFLOAD_FLAG=()
[ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && OFFLOAD_FLAG=(--cpu-offload-gb "$OFFLOAD_GB")  # stage 3

echo "[serve-sglang] launch_server $MODEL  TP=$TP  radix-cache(prefix)=ON  offload_gb=$OFFLOAD_GB"
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name minimax \
  --tp "$TP" \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "$MAX_MODEL_LEN" \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" "${OFFLOAD_FLAG[@]}" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

echo "[serve-sglang] waiting for /health on :$PORT ..."
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[serve-sglang] healthy"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[serve-sglang] server died during startup" >&2; exit 1; fi
  sleep 10
done

python3 -m pip install --break-system-packages -q -r "$HERE/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$HERE/requirements.txt" || true

# GPU hardware telemetry for the whole sweep (1 Hz)
echo "[serve-sglang] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $SRV $GPUMON 2>/dev/null || true' EXIT

echo "[serve-sglang] starting sweep -> $RESULT_DIR"
cd "$HERE"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:$PORT" \
  --metrics-url "http://localhost:$PORT/metrics" \
  --model minimax --n-gpu "$TP" \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 agg SGLang TP$TP offload=${OFFLOAD_GB}GB"

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true
echo "[serve-sglang] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
