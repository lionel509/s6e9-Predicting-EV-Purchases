#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain2[01234]" > /dev/null; do sleep 15; done
for s in "v34_init_score.py 10 42 k_inc100" "v34_init_score.py 10 42 k_inc1000"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain25 done" >> runs.log
