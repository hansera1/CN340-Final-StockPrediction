"""E25: ราคาจากแต่ละแบบ แทนค่าเมื่อไม่มีผล และเกณฑ์ผ่าน"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT
from forecast.evaluation.metrics import tick_size


@pytest.fixture(scope="module")
def e25():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("52_e25_all_flat")


class Rule:
    @staticmethod
    def price_rule(p_ref, p_eq, p_price):
        up, dn = (1 - p_eq) * p_price, (1 - p_eq) * (1 - p_price)
        shift = np.where(up > 0.5, 1, np.where(dn > 0.5, -1, 0))
        return p_ref + shift * tick_size(p_ref), shift


def move(f, u, up, dn):
    return (1 - f) * (u * up + (1 - u) * dn)


def frame():
    k = {"date": ["a", "b"], "symbol": ["KBANK", "KBANK"], "session": ["morning", "morning"]}
    x = pd.DataFrame({**k, "p_ref": [245.0, 245.0], "p_flat": [0.5, 0.2], "p_ud": [0.5, 0.7], "r_up": [0.02, 0.02], "r_dn": [-0.01, -0.01]})
    flat = pd.DataFrame({"date": ["a"], "symbol": ["KBANK"], "session": ["morning"], "m": [0.2]})
    up = pd.DataFrame({"date": ["a"], "symbol": ["KBANK"], "session": ["morning"], "m": [0.7]})
    return x, flat, up


def test_points_use_the_model_probabilities(e25):
    x, flat, up = frame()
    p, _ = e25.add_family(x, "F", flat, up, Rule, move)
    assert p.loc[0, "pt_F:m-rule"] == 246.0
    assert np.isclose(p.loc[0, "pt_F:m-mean"], 245.0 * np.exp(move(0.2, 0.7, 0.02, -0.01)))


def test_missing_rows_use_the_fallback_and_are_reported(e25):
    x, flat, up = frame()
    p, fill = e25.add_family(x, "F", flat, up, Rule, move)
    assert np.isclose(p.loc[1, "pt_F:m-mean"], 245.0 * np.exp(move(0.2, 0.7, 0.02, -0.01)))
    assert fill == [{"family": "F", "model": "m", "filled_share": 0.5}]


def test_verdict_needs_every_criterion(e25):
    rows = []
    for th in ("เช้า", "บ่าย"):
        for g in (th, f"KBANK {th}", f"ADVANC {th}"):
            rows.append({"model": "m", "group": g, "rmse": 1.0, "mae": 1.0, "d_rmse_pra": -0.1, "d_rmse_pra_lo": -0.2, "d_rmse_pra_hi": -0.05, "d_mae_pra": 0.0, "d_mae_pra_lo": -0.1})
    t = pd.DataFrame(rows)
    t.loc[t["group"] == "บ่าย", "d_mae_pra_lo"] = 0.01
    v = e25.verdict(t, ["m"]).set_index("session")["pass"]
    assert v["เช้า"] and not v["บ่าย"]
