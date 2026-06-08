#!/usr/bin/env bash
# Poll all colocated + disagg jobs and their result files. Appends to watch.log.
# Stop with: kill $(cat ~/gpumode-triton/results/watch.pid)
set -u
RES=~/gpumode-triton/results
LOG=$RES/watch.log
INTERVAL=${INTERVAL:-300}   # seconds between checks

COLO="dsv4_b1 kimi_b1 dsv4_b1_sglang kimi_b1_sglang"
DISAGG="dsv4_sglang_4p_4d dsv4_sglang_8p_4d dsv4_sglang_4p_8d dsv4_vllm_4p_4d dsv4_vllm_8p_4d dsv4_vllm_4p_8d"

echo $$ > "$RES/watch.pid"

while true; do
  TS=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
  {
    echo "================ $TS ================"
    echo "--- squeue ---"
    squeue -u ppbhatt500 -o "%.8i %.28j %.8T %.6D %.12M %.10l" 2>/dev/null || echo "squeue unavailable"
    echo "--- results (conc files + pareto) ---"
    for d in $COLO $DISAGG; do
      dir="$RES/$d"
      [ -d "$dir" ] || { printf "  %-22s (no dir)\n" "$d"; continue; }
      conc=$(ls "$dir"/conc*.json 2>/dev/null | wc -l | tr -d ' ')
      pareto="no"; [ -f "$dir/pareto.csv" ] && pareto="YES"
      prom="no"; [ -f "$dir/server_metrics.prom" ] && prom="yes"
      printf "  %-22s conc=%s pareto=%s prom=%s\n" "$d" "$conc" "$pareto" "$prom"
    done
    # flag any newly-finished paretos
    DONE=$(for d in $COLO $DISAGG; do [ -f "$RES/$d/pareto.csv" ] && echo "$d"; done | tr '\n' ' ')
    echo "--- pareto.csv present for: ${DONE:-none} ---"
    echo
  } >> "$LOG" 2>&1

  # exit early if no jobs left in the queue (-h strips the header row)
  NJOBS=$(squeue -u ppbhatt500 -h 2>/dev/null | grep -c .)
  if [ "${NJOBS:-0}" -eq 0 ]; then
    echo "[$TS] no jobs left in squeue — watcher exiting" >> "$LOG"
    break
  fi
  sleep "$INTERVAL"
done
