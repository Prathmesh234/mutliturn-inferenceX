#!/usr/bin/env bash
# Stage 1-3, DISAGG 1P1D (vLLM, NixlConnector): serve MiniMax-M2.5 DISAGGREGATED on one
# node — 1 prefill GPU (TP1, :8100) + 1 decode GPU (TP1, :8200). KV moves P->D via vLLM's
# NixlConnector (kv_role kv_both; UCX rides cuda_ipc/NVLink on-node), fronted by vLLM's
# OFFICIAL Nixl integration proxy (toy_proxy_server.py, vendored next to this script;
# OpenAI endpoint :8000). NixlConnector is the SUPPORTED P/D connector — P2pNcclConnector
# was experimental and deadlocked on the decode side (received KV but never generated),
# and InferenceX uses NixlConnector for all its vLLM disagg. Then the EXACT SAME sweep as
# aggregate. Prefix caching ON on both instances. EP does not apply here (TP1 per instance).
# Per-GPU numbers keep the harness n_gpu=4 denominator (= per quarter-node; 2 of 4 GPUs used).
#
# Submit via jobs/run_disagg_1p1d.sbatch (vLLM image).
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
PROXY="$HERE/toy_proxy_server.py"
RESULT_DIR="${RESULT_DIR:-$ROOT/../results_minimax/disagg_vllm_1p1d}"
mkdir -p "$RESULT_DIR"

# run.log = DECODE engine + sweep markers ONLY (what recompute_steady parses). The prefill
# engine's 'generation throughput .. Running:' lines (1-token prefills) would poison the
# steady medians -> prefill.log; proxy -> proxy.log.
RUN_LOG="$RESULT_DIR/run.log"; : > "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

# NixlConnector needs the nixl wheel (cu13 matches this image); the toy proxy needs httpx
# (fastapi/uvicorn already ship with vllm).
python3 -c "import nixl" 2>/dev/null || python3 -m pip install -q nixl-cu13 2>/dev/null || python3 -m pip install -q nixl
python3 -c "import httpx" 2>/dev/null || python3 -m pip install -q httpx

# Same engine flags as serve_agg.sh, minus EP (TP1) and minus CPU KV offload. NixlConnector
# with kv_role kv_both on BOTH instances (the proxy designates prefill vs decode per
# request); VLLM_NIXL_SIDE_CHANNEL_PORT must be unique per instance; UCX_TLS=all lets UCX
# pick cuda_ipc/NVLink on-node.
echo "[disagg-1p1d] prefill GPU0 :8100 (NixlConnector kv_both) -> prefill.log"
CUDA_VISIBLE_DEVICES=0 UCX_TLS=all UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve "$MODEL" \
  --served-model-name minimax \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  > "$RESULT_DIR/prefill.log" 2>&1 &
PRE=$!

echo "[disagg-1p1d] decode GPU1 :8200 (NixlConnector kv_both) -> run.log"
CUDA_VISIBLE_DEVICES=1 UCX_TLS=all UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve "$MODEL" \
  --served-model-name minimax \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
DEC=$!
trap 'kill $PRE $DEC 2>/dev/null || true' EXIT

echo "[disagg-1p1d] waiting for /health on :8100 and :8200 ..."
for PORT in 8100 8200; do
  for i in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[disagg-1p1d] :$PORT healthy"; break; fi
    if ! kill -0 $PRE $DEC 2>/dev/null; then echo "[disagg-1p1d] a server died during startup" >&2; exit 1; fi
    sleep 10
  done
done

# Official Nixl integration proxy: one OpenAI endpoint on :8000 -> prefiller :8100, decoder :8200.
echo "[disagg-1p1d] proxy (official nixl toy_proxy) :8000 -> P:8100 D:8200 -> proxy.log"
python3 "$PROXY" --host 0.0.0.0 --port 8000 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200 > "$RESULT_DIR/proxy.log" 2>&1 &
PXY=$!
trap 'kill $PRE $DEC $PXY 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8000/healthcheck" >/dev/null 2>&1; then echo "[disagg-1p1d] proxy healthy"; break; fi
  if ! kill -0 $PXY 2>/dev/null; then echo "[disagg-1p1d] proxy died (see proxy.log)" >&2; exit 1; fi
  sleep 2
done

# replayer deps in an isolated venv (matches the colocated launcher pattern)
python3 -m pip install --break-system-packages -q -r "$ROOT/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$ROOT/requirements.txt" || true

# GPU hardware telemetry for the whole sweep (power / util / temp / mem, 1 Hz).
echo "[disagg-1p1d] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $PRE $DEC $PXY $GPUMON 2>/dev/null || true' EXIT

# Sweep through the PROXY (:8000); prom ground truth from the DECODE server (:8200) —
# token counters/TPOT/E2E live where tokens are generated. Decode-side TTFT excludes
# prefill compute -> mildly optimistic, documented not hidden.
echo "[disagg-1p1d] starting sweep -> $RESULT_DIR"
cd "$ROOT"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:8000" \
  --metrics-url "http://localhost:8200/metrics" \
  --model minimax --n-gpu 4 \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 disagg vLLM 1P1D"
# prefill-side counters (prefix-cache hits live there in disagg) for reference
curl -s "http://localhost:8100/metrics" > "$RESULT_DIR/prefill.final.prom" || true

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true

# Make the metrics correct: recompute_steady swaps the raw whole-window throughput and
# bucket-coarse decode latencies for STEADY full-batch tput + interactivity (log-derived,
# running>=0.9*conc) in each conc<N>.json; then --replot rebuilds pareto.csv/png from the
# corrected JSONs (sweep wrote them mid-run, before recompute, so their steady cols are empty).
sync; sleep 3
echo "[disagg-1p1d] recompute_steady -> steady full-batch tput/interactivity from $RUN_LOG"
python3 recompute_steady.py --run-dir "$RESULT_DIR" --log "$RUN_LOG" \
  || echo "[disagg-1p1d] recompute_steady FAILED — rerun: python3 recompute_steady.py --run-dir '$RESULT_DIR' --log '$RUN_LOG'"
echo "[disagg-1p1d] replot pareto from corrected conc<N>.json"
python3 sweep.py --replot --dataset "$DATASET" --result-dir "$RESULT_DIR" \
  --title "MiniMax-M2.5 disagg vLLM 1P1D" \
  || echo "[disagg-1p1d] replot FAILED — rerun: python3 sweep.py --replot --dataset '$DATASET' --result-dir '$RESULT_DIR'"
echo "[disagg-1p1d] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
