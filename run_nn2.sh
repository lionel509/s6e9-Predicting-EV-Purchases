#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_nn1.sh" > /dev/null; do sleep 20; done
for s in "v35_realmlp.py 5 42 3"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs_nn.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a runs_nn.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs_nn.log
done
echo "=== nn2 done" >> runs_nn.log
