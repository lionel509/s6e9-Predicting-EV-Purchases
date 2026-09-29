#!/bin/zsh
# 2026-09-29 overnight CPU queue after the k20 ES check: more heuljax k20 seeds, then agg/init 20-fold seeds (issue #1)
cd "${0:A:h}"
while kill -0 6110 2>/dev/null; do sleep 60; done   # v41fix waiter (k20 fix1000 s7)
for s in 303 404 505 42; do
  echo "=== N_FOLDS=20 v40_heuljax.py $s start $(date +%T)"
  N_FOLDS=20 .venv/bin/python -u v40_heuljax.py $s 2>&1 | grep --line-buffered -v "^\["
  echo "=== exit $(date +%T)"
done
for c in "v37_hybrid_modes.py agg 20 7 k_inc100" "v37_hybrid_modes.py agg 20 2026 k_inc100" "v34_init_score.py 20 101 k_inc100"; do
  echo "=== $c start $(date +%T)"
  .venv/bin/python -u ${=c} 2>&1 | grep --line-buffered -v "^\["
  echo "=== exit $(date +%T)"
done
echo "=== chain34 done"
