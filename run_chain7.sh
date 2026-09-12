#!/bin/sh
cd "$(dirname "$0")"
for s in "v28_hybrid_plus.py --m1 --pseudo" "v27_hybrid.py 10 42" "v28_hybrid_plus.py --pseudo"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain7 done" >> runs.log
