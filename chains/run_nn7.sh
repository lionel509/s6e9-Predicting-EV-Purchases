#!/bin/sh
# 2026-09-27: RealMLP from the init100 margin (Fable idea 2), rerun after the LightGBM-before-torch import segfault fix
cd "$(dirname "$0")"
for s in 42 7 2026; do
  echo "=== models/v35_realmlp.py 10 $s 2 k_inc100 start $(date -u +%H:%M:%S)" >> logs/runs_nn.log
  .venv/bin/python models/v35_realmlp.py 10 $s 2 k_inc100 2>&1 | grep --line-buffered -v -i -E "warning|epoch|best score" | tee -a logs/runs_nn.log
  echo "=== exit $? $(date -u +%H:%M:%S)" >> logs/runs_nn.log
done
echo "=== nn7 done" >> logs/runs_nn.log
