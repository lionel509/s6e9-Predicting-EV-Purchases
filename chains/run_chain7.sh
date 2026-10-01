#!/bin/sh
cd "$(dirname "$0")"
for s in "models/v28_hybrid_plus.py --m1 --pseudo" "models/v27_hybrid.py 10 42" "models/v28_hybrid_plus.py --pseudo"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain7 done" >> logs/runs.log
