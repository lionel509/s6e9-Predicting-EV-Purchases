#!/bin/sh
cd "$(dirname "$0")"
for s in "models/v27_hybrid.py 10 303" "models/v27_hybrid.py 10 404" "models/v27_hybrid.py 10 505"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain13 done" >> logs/runs.log
