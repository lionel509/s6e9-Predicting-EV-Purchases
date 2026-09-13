#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain1[456789]|run_chain2[01]" > /dev/null; do sleep 15; done
for s in "v34_init_score.py 10 7 k_inc_exact" "v34_init_score.py 10 2026 k_inc_exact"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain22 done" >> runs.log
