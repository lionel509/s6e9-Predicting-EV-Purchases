#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_nn[12].sh" > /dev/null; do sleep 20; done
for s in "models/v35_realmlp.py 10 101" "models/v35_realmlp.py 10 202"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs_nn.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a logs/runs_nn.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs_nn.log
done
echo "=== nn3 done" >> logs/runs_nn.log
