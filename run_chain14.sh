#!/bin/sh
cd "$(dirname "$0")"
for s in "v32_hybrid_variants.py tecv 5 42 10" "v32_hybrid_variants.py wobble 5 42 2" "v32_hybrid_variants.py rank 5 42 8" "v32_hybrid_variants.py linear 5 42 0"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain14 done" >> runs.log
