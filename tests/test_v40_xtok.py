"""XTOK=1 adds GPT-2 income token keys (and token x subsidy/home/city crosses) to heuljax's exact-match MSTE rate keys.
v40 runs at import, so the pieces are exec'd from source."""
import os, re, sys
from pathlib import Path
import numpy as np

SRC = (Path(__file__).parent.parent / "v40_heuljax.py").read_text()


def chunk(start, end_pat):
    i = SRC.index(start); j = SRC.index(end_pat, i + len(start)); return SRC[i:j]


def ns(xtok):
    g = {"np": np, "os": os, "XTOK": xtok}
    exec(chunk("def mste_keys(raw):", "\nclass SourceIncome"), g)
    return g


RAW = {"Annual_Income_USD": np.array([92887.0, 30000.0, 30000.0, np.nan]), "Daily_Commute_km": np.array([23.4, 5.0, 5.0, 1.0]),
       "Subsidy_Available": np.array([0, 1, 0, 2], np.int16), "Home_Charging_Possible": np.array([0, 1, 1, 0], np.int16),
       "City_Type": np.array([0, 1, 2, 0], np.int16)}


def test_off_keeps_four_keys():
    assert list(ns(False)["mste_keys"](RAW)) == ["INC10", "INC100", "INC1K", "CMTINT"]


def test_on_adds_token_keys():
    k = ns(True)["mste_keys"](RAW)
    assert list(k)[:4] == ["INC10", "INC100", "INC1K", "CMTINT"]
    assert {"TOK1", "TOKLAST", "TOK1xSUB", "TOK1xHOME", "TOK1xCITY"} <= set(k)
    assert k["TOK1"][1] == k["TOK1"][2] and k["TOK1"][0] != k["TOK1"][1]      # same income, same token; different incomes differ
    assert k["TOK1xSUB"][1] != k["TOK1xSUB"][2]                                 # same token, different subsidy
    assert k["TOK1"][3] == -1 and all(np.isfinite(k[n]).all() for n in ("TOK1", "TOKLAST", "TOK1xSUB"))  # missing income gets its own code (old keys are median-filled upstream)


def test_keys_list_and_name_follow_switch():
    m = re.search(r"^MSTE_KEYS = .*$", SRC, re.M).group(0); g = {"XTOK": True}; exec(m, g)
    assert g["MSTE_KEYS"][4:] == ["TOK1", "TOKLAST", "TOK1xSUB", "TOK1xHOME", "TOK1xCITY"]
    line = re.search(r"^EXP_NAME = .*$", SRC, re.M).group(0)
    g = dict(K=10, sys=type("S", (), {"argv": ["x", "7"]}), FIXED_ROUNDS=0, DISTILL=0, HOLDOUT=False, XTOK=True); exec(line, g)
    assert g["EXP_NAME"] == "v40_heuljax_k10_s7_xtok"
