#!/bin/sh
cd "$(dirname "$0")"
for s in "v27_hybrid.py 10 303" "v27_hybrid.py 10 404" "v27_hybrid.py 10 505"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain13 done" >> runs.log
