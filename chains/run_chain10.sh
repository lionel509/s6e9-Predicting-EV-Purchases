#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain[78].sh" > /dev/null; do sleep 15; done
for s in "models/v27_hybrid.py 10 7" "models/v27_hybrid.py 10 2026"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain10 done" >> logs/runs.log
