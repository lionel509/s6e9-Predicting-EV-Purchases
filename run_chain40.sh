#!/bin/zsh
# 2026-09-29 night (issue #1): EXTRA=cross (+0.000015 at k10 s7) at k20 x 4 seeds -> group init100_tokx_k20
cd "${0:A:h}"
for s in 7 2026 101 42; do
  echo "=== cross k20 s$s start $(date +%T)"
  TOKENS=1 EXTRA=cross nice -n 10 .venv/bin/python -u v34_init_score.py 20 $s k_inc100 2>&1 | grep --line-buffered -E " OOF 0\.|Traceback|Error"
  echo "=== exit $(date +%T)"
done
echo "=== chain40 done $(date +%T)"
