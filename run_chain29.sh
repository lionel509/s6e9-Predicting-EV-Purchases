#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain2[0-8]" > /dev/null; do sleep 15; done
for s in "v33_hybrid_views.py C 10 42 lgb k_inc100"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain29 done" >> runs.log
