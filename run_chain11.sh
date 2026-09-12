#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain(9|10).sh" > /dev/null; do sleep 15; done
for s in "v27_hybrid.py 10 101" "v27_hybrid.py 10 202"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain11 done" >> runs.log
