#!/bin/bash
# Watch the 7 disagg jobs through model-load -> health -> benchmark, and CONFIRM
# Prometheus collection (server_metrics.prom appears + grows). Exits when every
# live job has a growing prom file, or after the time cap. Flags any job death.
RES=/mnt/home/ppbhatt500/gpumode-triton/results
JOBS="dsv4_vllm_4p_4d dsv4_vllm_8p_4d dsv4_vllm_4p_8d dsv4_vllm_4p_12d dsv4_sglang_4p_4d dsv4_sglang_4p_8d dsv4_sglang_4p_12d"
CAP=$((50*60)); t=0; INT=60
declare -A prevsz
while [ $t -lt $CAP ]; do
  echo "===== t=${t}s  $(date -u +%H:%M:%S) ====="
  squeue -u ppbhatt500 -h -o "%.8i %.22j %.9T %.6M %R" 2>/dev/null
  allprom=1
  for n in $JOBS; do
    d="$RES/$n"; pf="$d/server_metrics.prom"
    jid=$(grep -oE "jobid [0-9]+" "$RES/$n.driver.log" 2>/dev/null | head -1 | awk '{print $2}')
    alive=$(squeue -h -j "${jid:-0}" 2>/dev/null | wc -l)
    sz=$( [ -f "$pf" ] && stat -c%s "$pf" 2>/dev/null || echo 0 )
    grow=""; [ -n "${prevsz[$n]:-}" ] && [ "$sz" -gt "${prevsz[$n]}" ] && grow="GROWING"
    prevsz[$n]=$sz
    concs=$(ls "$d"/conc*.json 2>/dev/null | wc -l)
    state="alive=$alive prom=${sz}B $grow conc_done=$concs"
    if [ "$alive" = "0" ] && [ -n "$jid" ]; then state="*** JOB $jid GONE *** $state"; fi
    printf "  %-22s %s\n" "$n" "$state"
    [ "$sz" = "0" ] && allprom=0
  done
  if [ "$allprom" = "1" ]; then echo ">>> ALL 7 have server_metrics.prom — metrics collection CONFIRMED"; break; fi
  sleep $INT; t=$((t+INT))
done
echo "===== watcher done (t=${t}s) ====="
