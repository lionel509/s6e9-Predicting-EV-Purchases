"""EXTRA=km / EXTRA=cross add new TE keys to v27_hybrid.build (on top of TOKENS=1) and a matching suffix in v34's name;
with EXTRA unset, build's keys are unchanged, so every earlier run reproduces."""
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

ROWS = [(66, 92887.0, 23.4, 2, 3, 7, 1.0, "Male", "Suburban", "Sedan", "Yes", "No", "Low", "No"),
        (38, 30000.0, 5.0, 1, 2, 2, 4.0, "Female", "Rural", "SUV", "No", "Yes", "High", "Yes"),
        (45, 41250.0, 12.7, 3, 0, 5, 3.0, "Other", "Urban", "Hatchback", "Yes", "Yes", "Medium", "No"),
        (29, 170600.0, 40.0, 1, 4, 1, 5.0, "Male", "Urban", "Truck", "No", "No", "Low", "Yes")]
COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available",
        "Range_Anxiety_Level", "Will_Buy_EV"]


def keys(env):
    for k in ("TOKENS", "EXTRA"): os.environ.pop(k, None)
    os.environ.update(env)
    import v27_hybrid
    d = pd.DataFrame(ROWS, columns=COLS)
    _, _, K, Kte = v27_hybrid.build(d.iloc[:3], d.iloc[3:], d.drop(columns=[]))
    return K, Kte


def test_extra_unset_keeps_keys():
    K, _ = keys({"TOKENS": "1"})
    assert not any(c.startswith(("k_km_exact", "k_kmtok", "k_x_")) for c in K.columns)


def test_extra_km_adds_commute_keys():
    K, Kte = keys({"TOKENS": "1", "EXTRA": "km"})
    assert list(K["k_km_exact"]) == ["234", "50", "127"] and list(Kte["k_km_exact"]) == ["400"]
    assert {"k_kmtok1", "k_kmtoklast"} <= set(K.columns)


def test_extra_cross_adds_token_context_keys():
    K, _ = keys({"TOKENS": "1", "EXTRA": "cross"})
    for c in ("Subsidy_Available", "Home_Charging_Possible", "City_Type"):
        col = f"k_x_tok1_{c}"
        assert col in K.columns and K[col].iloc[1] == K["k_tok1"].iloc[1] + "|" + ROWS[1][COLS.index(c)]


def test_v34_name_carries_extra():
    src = (Path(__file__).parent.parent / "v34_init_score.py").read_text()
    line = next(l for l in src.splitlines() if l.startswith("name = "))
    ns = dict(os=os, KEY="k_inc100", DROP=False, XGB=False, N=10, SEED=7, SMOKE=False)
    os.environ.update({"TOKENS": "1", "EXTRA": "km"}); exec(line, ns)
    assert ns["name"] == "v34_init_inc100_tok_km_k10_s7"
    os.environ.pop("EXTRA"); exec(line, ns)
    assert ns["name"] == "v34_init_inc100_tok_k10_s7"


def test_extra_cross2_adds_more_crosses():
    K, _ = keys({"TOKENS": "1", "EXTRA": "cross2"})
    for c in ("Subsidy_Available", "Home_Charging_Possible", "City_Type", "Range_Anxiety_Level", "Environmental_Concern_Level", "Current_Car_Type"):
        assert f"k_x_tok1_{c}" in K.columns
    for c in ("Subsidy_Available", "Home_Charging_Possible", "City_Type"):
        assert K[f"k_x_toklast_{c}"].iloc[0] == K["k_toklast"].iloc[0] + "|" + ROWS[0][COLS.index(c)]


def test_extra_cross3_adds_tok2_crosses():
    K, _ = keys({"TOKENS": "1", "EXTRA": "cross3"})
    for c in ("Subsidy_Available", "Home_Charging_Possible", "City_Type"):
        assert f"k_x_tok1_{c}" in K.columns
        assert K[f"k_x_tok2_{c}"].iloc[2] == K["k_tok2"].iloc[2] + "|" + ROWS[2][COLS.index(c)]
    assert "k_x_toklast_City_Type" not in K.columns
