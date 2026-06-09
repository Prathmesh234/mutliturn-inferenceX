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
# Env knobs: MODEL, TP, MAX_MODEL_LEN, CONCURRENCIES, PREARRANGE_FRAC, RAMP_SECONDS,
#   DATASET, RESULT_DIR, OFFLOAD_GB (CPU KV-cache offload, GB, via HiCache --hicache-size).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL:?set MODEL to the MiniMax-M2.5 path}"
TP="${TP:-4}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"; OFFLOAD_GB="${OFFLOAD_GB:-16}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16,32,64,128}"; PREARRANGE_FRAC="${PREARRANGE_FRAC:-0.75}"
RAMP_SECONDS="${RAMP_SECONDS:-60}"; DATASET="${DATASET:-$HERE/batch_long.replay.jsonl}"
PORT="${PORT:-8000}"
ENGINE_TAG="sglang_tp${TP}"; { [ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && ENGINE_TAG="${ENGINE_TAG}_hicache${OFFLOAD_GB}gb"; } || true
RESULT_DIR="${RESULT_DIR:-$HERE/../results_minimax/agg_${ENGINE_TAG}}"
mkdir -p "$RESULT_DIR"

# Tee engine logs + sweep '===== concurrency N' markers into ONE run.log so the
# recompute_steady step (end of script) can segment phases and extract STEADY full-batch
# throughput + interactivity. The server (launched below) inherits these fds, so its
# periodic #running-req / gen-throughput lines land here interleaved with the markers.
RUN_LOG="$RESULT_DIR/run.log"; : > "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

# CPU KV-CACHE offload via HiCache: --enable-hierarchical-cache adds a host(CPU) KV tier
# (HiRadixCache L2); --hicache-size GB sets that host pool to an ABSOLUTE size in GB
# (overrides the default --hicache-ratio) -> apples-to-apples with vLLM's 16 GB
# --kv-offloading-size. SGLang's ONLY KV->CPU path (--cpu-offload-gb is model WEIGHTS).
# OFFLOAD_GB=0 disables. Verify once: python3 -m sglang.launch_server --help | grep -i hicache
HICACHE_FLAG=()
[ "$OFFLOAD_GB" -gt 0 ] 2>/dev/null && HICACHE_FLAG=(--enable-hierarchical-cache --hicache-size "$OFFLOAD_GB")

echo "[serve-sglang] launch_server $MODEL  TP=$TP  radix-cache(prefix)=ON  hicache_size_gb=$OFFLOAD_GB"
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name minimax \
  --tp "$TP" \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "$MAX_MODEL_LEN" \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" "${HICACHE_FLAG[@]}" &
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
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 agg SGLang TP$TP hicache=${OFFLOAD_GB}GB"

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true

# Make the metrics correct: recompute_steady swaps the raw whole-window throughput and
# bucket-coarse decode latencies for STEADY full-batch tput + interactivity (log-derived,
# running>=0.9*conc) in each conc<N>.json; then --replot rebuilds pareto.csv/png from the
# corrected JSONs (sweep wrote them mid-run, before recompute, so their steady cols are empty).
sync; sleep 3
echo "[serve-sglang] recompute_steady -> steady full-batch tput/interactivity from $RUN_LOG"
python3 recompute_steady.py --run-dir "$RESULT_DIR" --log "$RUN_LOG" \
  || echo "[serve-sglang] recompute_steady FAILED — rerun: python3 recompute_steady.py --run-dir '$RESULT_DIR' --log '$RUN_LOG'"
echo "[serve-sglang] replot pareto from corrected conc<N>.json"
python3 sweep.py --replot --dataset "$DATASET" --result-dir "$RESULT_DIR" \
  --title "MiniMax-M2.5 agg SGLang TP$TP hicache=${OFFLOAD_GB}GB" \
  || echo "[serve-sglang] replot FAILED — rerun: python3 sweep.py --replot --dataset '$DATASET' --result-dir '$RESULT_DIR'"
echo "[serve-sglang] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
