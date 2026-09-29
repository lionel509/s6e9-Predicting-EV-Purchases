#!/bin/zsh
# 2026-09-29: 30 folds (k30 s7 0.946374 vs k20 0.946333): heuljax seeds + k30 ES check, then agg k30 seeds (issue #1)
cd "${0:A:h}"
for c in "N_FOLDS=30 v40_heuljax.py 2026" "N_FOLDS=30 FIXED_ROUNDS=1000 v40_heuljax.py 7" "N_FOLDS=30 v40_heuljax.py 101" "N_FOLDS=30 v40_heuljax.py 202" \
         "v37_hybrid_modes.py agg 30 7 k_inc100" "v37_hybrid_modes.py agg 30 2026 k_inc100" "v37_hybrid_modes.py agg 30 42 k_inc100"; do
  echo "=== $c start $(date +%T)"
  a=(${=c}); envs=(${a:#*.py*}); envs=(${envs:#<->}); envs=(${(M)a:#*=*}); rest=(${a:#*=*})
  env $envs .venv/bin/python -u $rest 2>&1 | grep --line-buffered -E "OOF|Traceback|Error|FOLD .*AUC|fold [0-9]+:"
  echo "=== exit $(date +%T)"
done
echo "=== chain35 done"
