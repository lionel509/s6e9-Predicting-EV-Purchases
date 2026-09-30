#!/bin/sh
# 2026-09-27: seeds for the two new members blend_v26fr weights (agg_init100 0.13, init100_k20 0.21)
cd "$(dirname "$0")"
for s in "models/v37_hybrid_modes.py agg 10 7 k_inc100" "models/v37_hybrid_modes.py agg 10 2026 k_inc100" "models/v34_init_score.py 20 2026 k_inc100" "models/v37_hybrid_modes.py agg 20 42 k_inc100"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
echo "=== chain32 done" >> logs/runs.log
