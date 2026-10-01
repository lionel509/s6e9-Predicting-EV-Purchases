#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain(8|10).sh" > /dev/null; do sleep 15; done
for s in "models/v29_hybrid_cat.py 42 10" "models/v30_hybrid_xgb.py 42 10"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain9 done" >> logs/runs.log
