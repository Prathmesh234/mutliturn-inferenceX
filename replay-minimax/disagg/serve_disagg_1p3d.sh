#!/usr/bin/env bash
# Stage 1-3, DISAGG 1P3D (vLLM, NixlConnector): serve MiniMax-M2.5 DISAGGREGATED on one
# node — 1 prefill GPU (TP1, :8100) + 3 DECODE WORKERS (TP1 each, GPUs 1/2/3 on
# :8200/:8201/:8202). KV moves P->D via vLLM's NixlConnector (kv_role kv_both; UCX rides
# cuda_ipc/NVLink on-node); vLLM's OFFICIAL Nixl integration proxy (toy_proxy_server.py,
# vendored next to this script; OpenAI endpoint :8000) round-robins across the 3 decoders.
# NixlConnector is the SUPPORTED P/D connector (P2pNcclConnector deadlocked on decode;
# InferenceX uses Nixl for all vLLM disagg). Then the EXACT SAME sweep as aggregate.
#
# Why 3x TP1 and not one TP3 decode instance: MiniMax-M2.5 has 8 KV heads, 8 % 3 != 0
# -> TP3 cannot shard attention. EP does not apply (TP1 per instance, like 1P1D).
# Prefix caching ON everywhere; the long-prefix reuse lives on the SINGLE prefill instance.
# CAVEATS: sweep scrapes decode #1 only (~1/3 of traffic; counters undercount 3x);
# recompute_steady's full-batch filter (running >= 0.9*conc) never fires with 3 interleaved
# decode engines -> it WARNs and conc<N>.json keeps the whole-window prom numbers. Documented.
#
# Submit via jobs/run_disagg_1p3d.sbatch (vLLM image).
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
RESULT_DIR="${RESULT_DIR:-$ROOT/../results_minimax/disagg_vllm_1p3d}"
mkdir -p "$RESULT_DIR"

# run.log = the 3 DECODE engines + sweep markers (see caveats above); prefill -> prefill.log.
RUN_LOG="$RESULT_DIR/run.log"; : > "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

# NixlConnector needs the nixl wheel (cu13 matches this image); the toy proxy needs httpx.
python3 -c "import nixl" 2>/dev/null || python3 -m pip install -q nixl-cu13 2>/dev/null || python3 -m pip install -q nixl
python3 -c "import httpx" 2>/dev/null || python3 -m pip install -q httpx

# Same engine flags as serve_agg.sh, minus EP (TP1) and minus CPU KV offload. NixlConnector
# kv_role kv_both on all 4 instances; one VLLM_NIXL_SIDE_CHANNEL_PORT per instance.
echo "[disagg-1p3d] prefill GPU0 :8100 (NixlConnector kv_both) -> prefill.log"
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

echo "[disagg-1p3d] decode1 GPU1 :8200 (NixlConnector kv_both) -> run.log"
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
DEC1=$!

echo "[disagg-1p3d] decode2 GPU2 :8201 (NixlConnector kv_both) -> run.log"
CUDA_VISIBLE_DEVICES=2 UCX_TLS=all UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5602 \
vllm serve "$MODEL" \
  --served-model-name minimax \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 --port 8201 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
DEC2=$!

echo "[disagg-1p3d] decode3 GPU3 :8202 (NixlConnector kv_both) -> run.log"
CUDA_VISIBLE_DEVICES=3 UCX_TLS=all UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5603 \
vllm serve "$MODEL" \
  --served-model-name minimax \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 --port 8202 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
DEC3=$!
trap 'kill $PRE $DEC1 $DEC2 $DEC3 2>/dev/null || true' EXIT

echo "[disagg-1p3d] waiting for /health on :8100 :8200 :8201 :8202 ..."
for PORT in 8100 8200 8201 8202; do
  for i in $(seq 1 360); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[disagg-1p3d] :$PORT healthy"; break; fi
    if ! kill -0 $PRE $DEC1 $DEC2 $DEC3 2>/dev/null; then echo "[disagg-1p3d] a server died during startup" >&2; exit 1; fi
    sleep 10
  done
done

# Official Nixl integration proxy: one OpenAI endpoint on :8000 -> prefiller :8100,
# decoders :8200/:8201/:8202 (round-robin).
echo "[disagg-1p3d] proxy (official nixl toy_proxy) :8000 -> P:8100 D:8200,8201,8202 -> proxy.log"
python3 "$PROXY" --host 0.0.0.0 --port 8000 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost localhost localhost --decoder-ports 8200 8201 8202 \
  > "$RESULT_DIR/proxy.log" 2>&1 &
PXY=$!
trap 'kill $PRE $DEC1 $DEC2 $DEC3 $PXY 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8000/healthcheck" >/dev/null 2>&1; then echo "[disagg-1p3d] proxy healthy"; break; fi
  if ! kill -0 $PXY 2>/dev/null; then echo "[disagg-1p3d] proxy died (see proxy.log)" >&2; exit 1; fi
  sleep 2
done

# replayer deps in an isolated venv (matches the colocated launcher pattern)
python3 -m pip install --break-system-packages -q -r "$ROOT/requirements.txt" 2>/dev/null || \
  python3 -m pip install -q -r "$ROOT/requirements.txt" || true

# GPU hardware telemetry for the whole sweep (power / util / temp / mem, 1 Hz).
echo "[disagg-1p3d] GPU telemetry -> $RESULT_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>/dev/null &
GPUMON=$!
trap 'kill $PRE $DEC1 $DEC2 $DEC3 $PXY $GPUMON 2>/dev/null || true' EXIT

# Sweep through the PROXY (:8000); prom from decode #1 (:8200, ~1/3 of traffic — see
# header caveats). All decoders + prefill snapshot their final /metrics afterwards.
echo "[disagg-1p3d] starting sweep -> $RESULT_DIR"
cd "$ROOT"
python3 sweep.py \
  --dataset "$DATASET" \
  --base-url "http://localhost:8000" \
  --metrics-url "http://localhost:8200/metrics" \
  --model minimax --n-gpu 4 \
  --concurrencies "$CONCURRENCIES" \
  --prearrange-frac "$PREARRANGE_FRAC" --ramp-seconds "$RAMP_SECONDS" \
  --result-dir "$RESULT_DIR" --title "MiniMax-M2.5 disagg vLLM 1P3D"
curl -s "http://localhost:8100/metrics" > "$RESULT_DIR/prefill.final.prom" || true
curl -s "http://localhost:8200/metrics" > "$RESULT_DIR/decode1.final.prom" || true
curl -s "http://localhost:8201/metrics" > "$RESULT_DIR/decode2.final.prom" || true
curl -s "http://localhost:8202/metrics" > "$RESULT_DIR/decode3.final.prom" || true

kill $GPUMON 2>/dev/null || true
python3 gpu_metrics.py "$RESULT_DIR/gpu.csv" "$RESULT_DIR/gpu.json" >/dev/null 2>&1 || true

# recompute_steady will WARN here (3 interleaved decode engines never satisfy the
# single-engine full-batch filter) and keep the window averages — expected, see header.
sync; sleep 3
echo "[disagg-1p3d] recompute_steady (expect 'no steady sample' WARNs — see header)"
python3 recompute_steady.py --run-dir "$RESULT_DIR" --log "$RUN_LOG" \
  || echo "[disagg-1p3d] recompute_steady FAILED — rerun: python3 recompute_steady.py --run-dir '$RESULT_DIR' --log '$RUN_LOG'"
echo "[disagg-1p3d] replot pareto from conc<N>.json"
python3 sweep.py --replot --dataset "$DATASET" --result-dir "$RESULT_DIR" \
  --title "MiniMax-M2.5 disagg vLLM 1P3D" \
  || echo "[disagg-1p3d] replot FAILED — rerun: python3 sweep.py --replot --dataset '$DATASET' --result-dir '$RESULT_DIR'"
echo "[disagg-1p3d] sweep done. prom metrics in conc*.json/pareto.csv ; GPU metrics in gpu.json"
