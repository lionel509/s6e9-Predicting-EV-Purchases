#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "run_chain2[0-7]" > /dev/null; do sleep 15; done
for s in "v33_hybrid_views.py D 10 7 lgb k_inc50" "v33_hybrid_views.py D 10 2026 lgb k_inc50" "v33_hybrid_views.py C 10 42 lgb k_inc100" "v34_init_score.py 10 303 k_inc100" "v34_init_score.py 10 404 k_inc100" "v34_init_score.py 10 505 k_inc100"; do
  echo "=== $s start $(date -u +%H:%M:%S)" >> runs.log
  .venv/bin/python $s 2>&1 | grep --line-buffered -v -i -E "warning|eval_set" | tee -a runs.log
  echo "=== exit $(date -u +%H:%M:%S)" >> runs.log
done
echo "=== chain28 done" >> runs.log
