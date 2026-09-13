#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_nn[123].sh" > /dev/null; do sleep 20; done
for s in "v35_realmlp.py 10 303" "v35_realmlp.py 10 404"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs_nn.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a runs_nn.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs_nn.log
done
echo "=== nn4 done" >> runs_nn.log
