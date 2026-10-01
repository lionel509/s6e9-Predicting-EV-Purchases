#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain2[0-9]|run_chain3[1-9]" > /dev/null; do sleep 15; done
for s in "models/v34_init_score.py 20 42 k_inc100" "models/v34_init_score.py 20 7 k_inc100" "models/v32_hybrid_variants.py additive 10 42" "models/v34_init_score.py 10 42 k_inc100 xgb" "models/v34_init_score.py 10 42 k_km_int" "models/v34_init_score.py 10 42 k_inc100+k_km_int" "models/v32_hybrid_variants.py additive 10 42 k_inc100" "blend/refit_full.py init100 42 7 2026 101 202 303 404 505"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> logs/runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a logs/runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> logs/runs.log
done
.venv/bin/python blend/assemble.py blend_v25 2>&1 | tee logs/assemble_v25.out
echo "=== chain30 done" >> logs/runs.log
