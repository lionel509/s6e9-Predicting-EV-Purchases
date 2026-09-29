#!/bin/zsh
# 2026-09-29 resume step 3 (issue #1): own seeds of the GPT-2-token LR at 20 folds only (k30 dropped for time). Starts after chain37.
cd "${0:A:h}"
while kill -0 68684 2>/dev/null; do sleep 30; done
for s in 7 2026 101 202; do
  echo "=== v42 k20 s$s start $(date +%T)"
  N_FOLDS=20 nice -n 10 .venv/bin/python -u v42_hjlr.py $s 2>&1 | grep --line-buffered -E "OOF 0\.|Traceback|Error"
  echo "=== exit $(date +%T)"
done
echo "=== chain38 done $(date +%T)"
