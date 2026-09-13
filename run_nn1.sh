#!/bin/sh
# neural chain: runs on MPS, concurrently with the CPU chains; logs to runs_nn.log
cd "$(dirname "$0")"
for s in "v35_realmlp.py 5 42" "v35_realmlp.py 10 42" "v35_realmlp.py 10 7" "v35_realmlp.py 10 2026"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs_nn.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a runs_nn.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs_nn.log
done
echo "=== nn1 done" >> runs_nn.log
