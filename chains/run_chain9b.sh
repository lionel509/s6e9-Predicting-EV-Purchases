#!/bin/sh
cd "$(dirname "$0")"
for s in "models/v31_hybrid_nn.py 20 42"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain9b done" >> logs/runs.log
