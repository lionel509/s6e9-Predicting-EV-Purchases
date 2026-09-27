#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain2[0-9]|run_chain3[1-9]" > /dev/null; do sleep 15; done
for s in "v34_init_score.py 20 42 k_inc100" "v34_init_score.py 20 7 k_inc100" "v32_hybrid_variants.py additive 10 42" "v34_init_score.py 10 42 k_inc100 xgb" "v34_init_score.py 10 42 k_km_int" "v34_init_score.py 10 42 k_inc100+k_km_int" "v32_hybrid_variants.py additive 10 42 k_inc100" "refit_full.py init100 42 7 2026 101 202 303 404 505"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
.venv/bin/python assemble.py blend_v25 2>&1 | tee assemble_v25.out
echo "=== chain30 done" >> runs.log
