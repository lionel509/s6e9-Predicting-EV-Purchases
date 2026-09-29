#!/bin/zsh
# 2026-09-29 resume step 1 (issue #1): init100 + GPT-2 token keys at 20 folds, 4 seeds -> group init100_tok_k20
cd "${0:A:h}"
for s in 7 2026 101 42; do
  echo "=== tok k20 s$s start $(date +%T)"
  TOKENS=1 nice -n 10 .venv/bin/python -u v34_init_score.py 20 $s k_inc100 2>&1 | grep --line-buffered -E "OOF|Traceback|Error|AUC"
  echo "=== exit $(date +%T)"
done
echo "=== chain36 done"
