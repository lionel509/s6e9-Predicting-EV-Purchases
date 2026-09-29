"""v37's output name must carry _tok when TOKENS=1, like v34, or a token run overwrites the plain run's OOF."""
import os, re, sys
from pathlib import Path

SRC = (Path(__file__).parent.parent / "v37_hybrid_modes.py").read_text()
HEAD = SRC[SRC.index("MODE = sys.argv"):]
HEAD = HEAD[:HEAD.index("\n", HEAD.index("name = "))]


def name_for(argv, tokens):
    env = {k: v for k, v in os.environ.items() if k not in ("TOKENS", "SMOKE")}
    if tokens:
        env["TOKENS"] = "1"
    old_argv, old_env = sys.argv, dict(os.environ)
    try:
        sys.argv = ["v37_hybrid_modes.py", *argv]
        os.environ.clear(); os.environ.update(env)
        ns = {"os": os, "sys": sys}
        exec(HEAD, ns)
        return ns["name"]
    finally:
        sys.argv = old_argv; os.environ.clear(); os.environ.update(old_env)


def test_plain_name_unchanged():
    assert name_for(["agg", "20", "7", "k_inc100"], False) == "v37_agg_initinc100_k20_s7"


def test_token_name_has_tok():
    assert name_for(["agg", "20", "7", "k_inc100"], True) == "v37_agg_initinc100_tok_k20_s7"
