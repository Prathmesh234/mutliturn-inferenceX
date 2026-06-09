#!/bin/bash
RES=/mnt/home/ppbhatt500/gpumode-triton/results
SG="dsv4_sglang_4p_4d dsv4_sglang_4p_8d dsv4_sglang_4p_12d"
CAP=$((30*60)); t=0
while [ $t -lt $CAP ]; do
  echo "===== t=${t}s $(date -u +%H:%M:%S) ====="
  healthy=0
  for n in $SG; do
    jid=$(grep -oE "jobid [0-9]+" $RES/$n.driver.log 2>/dev/null | head -1 | awk '{print $2}')
    alive=$(squeue -h -j "${jid:-0}" 2>/dev/null | wc -l)
    h=$(grep -c "Server is healthy" $RES/$n.driver.log 2>/dev/null)
    err=$(grep -ciE "scheduler died|exit code: -|RuntimeError|CUDA error|KVTransfer|OOM|Traceback" $RES/$n.driver.log 2>/dev/null)
    pf=$RES/$n/server_metrics.prom; sz=$([ -f $pf ] && stat -c%s $pf || echo 0)
    [ "$h" -ge 1 ] && healthy=$((healthy+1))
    flag=""; [ "$alive" = "0" ] && [ -n "$jid" ] && flag="*** GONE ***"
    printf "  %-22s alive=%s healthy=%s err=%s prom=%sB %s\n" "$n" "$alive" "$h" "$err" "$sz" "$flag"
  done
  if [ "$healthy" = "3" ]; then echo ">>> all 3 sglang PASSED health-check with radix cache ON (init OK)"; break; fi
  sleep 45; t=$((t+45))
done
echo "===== sglang watcher done (t=${t}s) ====="
