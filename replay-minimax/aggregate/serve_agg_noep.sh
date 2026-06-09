#!/usr/bin/env bash
# Stage 1-3, vLLM NO-EXPERT-PARALLEL variant: serve MiniMax-M2.5 AGGREGATED
# (colocated prefill+decode, one node), then run the EXACT SAME sweep. This is the
# control for the EP comparison: the ONLY difference from serve_agg.sh is that
# `--enable-expert-parallel` is REMOVED, so the MoE runs pure tensor-parallel (matching
# the TP-only SGLang control). Lets us isolate EP as the single variable across the 2x2.
# Results land in a distinct agg_vllm_tp${TP}_noep/ dir.
#
# Env knobs (identical to serve_agg.sh): MODEL, TP, MAX_MODEL_LEN, OFFLOAD_GB,
#   CONCURRENCIES, PREARRANGE_FRAC, RAMP_SECONDS, DATASET, RESULT_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL:?set MODEL to the MiniMax-M2.5 path}"
TP="${TP:-4}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"; OFFLOAD_GB="${OFFLOAD_GB:-16}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"; PREARRANGE_FRAC="${PREARRANGE_FRAC:-0.75}"
RAMP_SECONDS="${RAMP_SECONDS:-60}"; DATASET="${DATASET:-$HERE/batch_long.replay.jsonl}"
PORT="${PORT:-8000}"
ENGINE_TAG="vllm_tp${TP}_noep"; { [ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && ENGINE_TAG="${ENGINE_TAG}_kvoff${OFFLOAD_GB}"; } || true
RESULT_DIR="${RESULT_DIR:-$HERE/../results_minimax/agg_${ENGINE_TAG}}"
mkdir -p "$RESULT_DIR"

# Tee engine logs + sweep '===== concurrency N' markers into ONE run.log so the
# recompute_steady step (end of script) can segment phases and extract STEADY full-batch
# throughput + interactivity. The server (launched below) inherits these fds, so its
# periodic throughput/running-batch lines land here interleaved with the markers.
RUN_LOG="$RESULT_DIR/run.log"; : > "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

OFFLOAD_FLAG=()
# CPU KV-CACHE offload (NOT model weights). vLLM native offloading connector backs up GPU
# KV blocks to pinned host RAM, reusable from CPU. VERIFY once on a node that this image
# has the flag:  vllm serve --help | grep -i offloading
# Fallback for builds without it: --kv-transfer-config '{"kv_connector":"OffloadingConnector",
#   "kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":<N>}}'
[ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && OFFLOAD_FLAG=(--kv-offloading-backend native --kv-offloading-size "$OFFLOAD_GB")

echo "[serve-noep] vllm serve $MODEL  TP=$TP  prefix-caching=ON  expert-parallel=OFF  kv_offload_gb=$OFFLOAD_GB"
vllm serve "$MODEL" \
  --served-model-name minimax \
  --tensor-parallel-size "$TP" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --port "$PORT" "${OFFLOAD_FLAG[@]}" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

echo "[serve-noep] waiting for /health on :$PORT ..."
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[serve-noep] healthy"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[serve-noep] server died during startup" >&2; exit 1; fi
  sleep 10
done

# replayer deps in an isolated venv (matches the colocated launcher pattern)
python3 -m pip install --break-system-packages -q -r "$HERE/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$HERE/requirements.txt" || true

# GPU hardware telemetry for the whole sweep (power / util / temp / mem, 1 Hz).
echo "[serve-noep] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $SRV $GPUMON 2>/dev/null || true' EXIT

echo "[serve-noep] starting sweep -> $RESULT_DIR"
cd "$HERE"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:$PORT" \
  --metrics-url "http://localhost:$PORT/metrics" \
  --model minimax --n-gpu "$TP" \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 agg vLLM TP$TP no-EP offload=${OFFLOAD_GB}GB"

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true

# Make the metrics correct: recompute_steady swaps the raw whole-window throughput and
# bucket-coarse decode latencies for STEADY full-batch tput + interactivity (log-derived,
# running>=0.9*conc) in each conc<N>.json; then --replot rebuilds pareto.csv/png from the
# corrected JSONs (sweep wrote them mid-run, before recompute, so their steady cols are empty).
sync; sleep 3
echo "[serve-noep] recompute_steady -> steady full-batch tput/interactivity from $RUN_LOG"
python3 recompute_steady.py --run-dir "$RESULT_DIR" --log "$RUN_LOG" \
  || echo "[serve-noep] recompute_steady FAILED — rerun: python3 recompute_steady.py --run-dir '$RESULT_DIR' --log '$RUN_LOG'"
echo "[serve-noep] replot pareto from corrected conc<N>.json"
python3 sweep.py --replot --dataset "$DATASET" --result-dir "$RESULT_DIR" \
  --title "MiniMax-M2.5 agg vLLM TP$TP no-EP offload=${OFFLOAD_GB}GB" \
  || echo "[serve-noep] replot FAILED — rerun: python3 sweep.py --replot --dataset '$DATASET' --result-dir '$RESULT_DIR'"
echo "[serve-noep] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
