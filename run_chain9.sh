#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain[78].sh" > /dev/null; do sleep 15; done
for s in "v29_hybrid_cat.py 42" "v30_hybrid_xgb.py 42"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain9 done" >> runs.log
