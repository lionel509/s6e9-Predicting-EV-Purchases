#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain2[012345]" > /dev/null; do sleep 15; done
for s in "models/v34_init_score.py 10 7 k_inc100" "models/v34_init_score.py 10 2026 k_inc100" "models/v34_init_score.py 10 101 k_inc100" "models/v34_init_score.py 10 202 k_inc100"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain26 done" >> logs/runs.log
