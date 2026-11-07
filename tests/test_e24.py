"""E24: นิยามราคานิ่งเทียบราคาเปิด ค่าเฉลี่ยใช้เฉพาะวันก่อนหน้า และสูตรราคา 2 ขั้น"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT
from forecast.evaluation.metrics import tick_size
from forecast.models.e9 import target_rows


@pytest.fixture(scope="module")
def e24():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("51_e24_two_step")


class Rule:
    @staticmethod
    def price_rule(p_ref, p_eq, p_price):
        up, dn = (1 - p_eq) * p_price, (1 - p_eq) * (1 - p_price)
        shift = np.where(up > 0.5, 1, np.where(dn > 0.5, -1, 0))
        return p_ref + shift * tick_size(p_ref), shift


def test_flat_label_compares_end_with_session_open_not_reference():
    meta = pd.DataFrame({"direction": [0.0, 1.0], "p_end": [338.0, 338.0], "p_start": [338.0, 337.0], "p_ref": [340.0, 338.0]})
    _, y = target_rows(meta, "flat_dir")
    assert y.tolist() == [1, 0]


def test_expected_move_formula(e24):
    assert np.isclose(e24.expected_move(0.2, 0.75, 0.02, -0.01), 0.8 * (0.75 * 0.02 + 0.25 * -0.01))


def test_flat_certainty_means_no_move(e24):
    assert e24.expected_move(1.0, 0.9, 0.02, -0.01) == 0.0


def test_conditional_returns_use_only_earlier_days(e24):
    days = pd.bdate_range("2026-01-05", periods=4, tz="Asia/Bangkok")
    d = pd.DataFrame({"date": days, "symbol": "KBANK", "session": "morning", "p_start": 100.0, "p_end": [101.0, 99.0, 102.0, 500.0]})
    m = e24.cond_r(d)
    assert np.isnan(m["r_up"].iloc[0]) and np.isclose(m["r_up"].iloc[1], np.log(1.01))
    assert np.isclose(m["r_up"].iloc[3], np.mean([np.log(1.01), np.log(1.02)]))
    assert np.isclose(m["r_dn"].iloc[3], np.log(0.99))


def frame():
    return pd.DataFrame({"session": ["morning", "afternoon"], "p_ref": [245.0, 245.0], "g_hat": [0.01, 0.01], "p_flat": [0.2, 0.2], "p_ud": [0.7, 0.7], "r_up": [0.02, 0.02], "r_dn": [-0.01, -0.01]})


def test_gap_shift_applies_to_morning_only(e24):
    x = e24.add_points(frame(), Rule).set_index("session")
    move = 0.8 * (0.7 * 0.02 + 0.3 * -0.01)
    assert np.isclose(x.loc["morning", "pt_Sg-mean"], 245 * np.exp(0.01 + move))
    assert np.isnan(x.loc["afternoon", "pt_Sg-mean"]) and np.isnan(x.loc["afternoon", "pt_Sg-rule"])
    assert np.isclose(x.loc["afternoon", "pt_S0-mean"], 245 * np.exp(move))


def test_rule_moves_one_tick_from_expected_start(e24):
    x = e24.add_points(frame(), Rule).set_index("session")
    assert x.loc["morning", "pt_S0-rule"] == 246.0     # โอกาสขึ้น 0.56 > 0.5
    assert x.loc["morning", "pt_Sg-rule"] == 248.0     # 245·e^0.01 = 247.46 → 247 แล้ว +1


def test_verdict_uses_the_preregistered_representative(e24):
    rows = []
    for m, th in [("Sg-mean", "เช้า"), ("S0-mean", "บ่าย")]:
        for g in (th, f"KBANK {th}", f"ADVANC {th}"):
            rows.append({"model": m, "group": g, "d_rmse_pra_hi": -0.1, "d_rmse_pra_lo": -0.2, "d_mae_pra_lo": -0.1})
    t = pd.DataFrame(rows)
    t.loc[(t["model"] == "S0-mean") & (t["group"] == "บ่าย"), "d_rmse_pra_hi"] = 0.1
    v = e24.verdict(t).set_index("session")["pass"]
    assert v["เช้า"] and not v["บ่าย"]
