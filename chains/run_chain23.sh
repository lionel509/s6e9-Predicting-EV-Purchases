#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain2[012]" > /dev/null; do sleep 15; done
for s in "models/v33_hybrid_views.py D 10 101" "models/v33_hybrid_views.py D 10 202"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain23 done" >> logs/runs.log
