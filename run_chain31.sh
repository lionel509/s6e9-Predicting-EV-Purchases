#!/bin/sh
# 2026-09-27 Fable end-game list, built by MiMo: v38 init100 shapes (idea 5), v37 mono / agg (ideas 4, 7); then the fold-ranked blend (idea 1)
cd "$(dirname "$0")"
until grep -q "chain30 done" runs.log; do sleep 30; done
for s in "v38_init_shapes.py 10 42 k_inc100 wide" "v38_init_shapes.py 10 42 k_inc100 col5" "v37_hybrid_modes.py mono 10 42 k_inc100" "v37_hybrid_modes.py agg 10 42 k_inc100" "v38_init_shapes.py 10 42 k_inc100 slow"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
.venv/bin/python assemble.py blend_v26 2>&1 | tee assemble_v26.out
.venv/bin/python assemble_fr.py blend_v26fr 2>&1 | tee assemble_v26fr.out
echo "=== chain31 done" >> runs.log
