#!/usr/bin/env bash
# Stage 1-3: serve MiniMax-M2.5 AGGREGATED (colocated prefill+decode) on one node,
# then run the concurrency sweep against it. Meant to run INSIDE the vLLM container
# on a GPU node (see run_agg.sbatch). Prefix caching ON (agentic traffic reuses
# long shared prefixes — the existing recipes ship it OFF, which is wrong for us).
#
# Env knobs:
#   MODEL            model path (e.g. /mnt/vast/models/minimax-m2.5-nvfp4)   [required]
#   TP               tensor-parallel size / GPUs in the node            (default 4)
#   MAX_MODEL_LEN    context length (agentic ISL runs long)             (default 40960)
#   OFFLOAD_GB       >0 => CPU weight offload (stage 3)                  (default 0)
#   CONCURRENCIES    sweep points                                       (default 4,8,16,32,64,128)
#   PREARRANGE_FRAC  steady-state start fraction                        (default 0.75)
#   RAMP_SECONDS     arrival ramp                                       (default 60)
#   DATASET          *.replay.jsonl                                     (default ./batch_long.replay.jsonl)
#   RESULT_DIR       output dir                                         (default ./out)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL:?set MODEL to the MiniMax-M2.5 path}"
TP="${TP:-4}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"; OFFLOAD_GB="${OFFLOAD_GB:-0}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"; PREARRANGE_FRAC="${PREARRANGE_FRAC:-0.75}"
RAMP_SECONDS="${RAMP_SECONDS:-60}"; DATASET="${DATASET:-$HERE/batch_long.replay.jsonl}"
PORT="${PORT:-8000}"
# Results land in a dedicated results_minimax/ folder, one subdir per config.
ENGINE_TAG="vllm_tp${TP}"; { [ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && ENGINE_TAG="${ENGINE_TAG}_offload${OFFLOAD_GB}"; } || true
RESULT_DIR="${RESULT_DIR:-$HERE/../results_minimax/agg_${ENGINE_TAG}}"
mkdir -p "$RESULT_DIR"

OFFLOAD_FLAG=()
[ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && OFFLOAD_FLAG=(--cpu-offload-gb "$OFFLOAD_GB")  # stage 3

echo "[serve] vllm serve $MODEL  TP=$TP  prefix-caching=ON  offload_gb=$OFFLOAD_GB"
vllm serve "$MODEL" \
  --served-model-name minimax \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --port "$PORT" "${OFFLOAD_FLAG[@]}" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

echo "[serve] waiting for /health on :$PORT ..."
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[serve] healthy"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[serve] server died during startup" >&2; exit 1; fi
  sleep 10
done

# replayer deps in an isolated venv (matches the colocated launcher pattern)
python3 -m pip install --break-system-packages -q -r "$HERE/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$HERE/requirements.txt" || true

# GPU hardware telemetry for the whole sweep (power / util / temp / mem, 1 Hz).
echo "[serve] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $SRV $GPUMON 2>/dev/null || true' EXIT

echo "[serve] starting sweep -> $RESULT_DIR"
cd "$HERE"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:$PORT" \
  --metrics-url "http://localhost:$PORT/metrics" \
  --model minimax --n-gpu "$TP" \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 agg TP$TP offload=${OFFLOAD_GB}GB"

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true
echo "[serve] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
