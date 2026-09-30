#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f chains/run_chain7.sh > /dev/null; do sleep 15; done
for s in "models/v28_hybrid_plus.py --m1 --pseudo"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain8 done" >> logs/runs.log
