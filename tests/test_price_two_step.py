"""ราคา E24 ในระบบจริง: สูตรตรง backtest ประวัติใช้เฉพาะวันก่อนหน้า และเปิดด้วยสวิตช์เท่านั้น"""
import copy

import numpy as np
import pandas as pd
import pytest

from forecast.config import load_rules
from forecast.live.predict import cond_r_means, two_step_point
from forecast.live.slots import live_specs, lock_specs, two_step_enabled, two_step_specs

RULES = load_rules()
TZ = RULES["timezone"]


def enabled_rules():
    r = copy.deepcopy(RULES)
    r["live"]["price_two_step"] = {"enabled": True, "up": "E12/fd_lr"}
    return r


def test_switch_is_off_in_the_committed_rules():
    assert not two_step_enabled(RULES)
    assert "price_two_step_up" not in live_specs(RULES)


def test_enabled_rules_require_the_up_model_and_use_atc_bundles():
    r = enabled_rules()
    assert two_step_specs(r) == ("R4ATC/E9FLAT/adir_lr", "R4ATC/E12/fd_lr")
    assert live_specs(r)["price_two_step_up"] == "R4ATC/E12/fd_lr"
    assert "E12/fd_lr" in lock_specs(r)


def test_point_formula_matches_e24():
    move = (1 - 0.2) * (0.75 * 0.02 + 0.25 * -0.01)
    assert np.isclose(two_step_point(245.0, 0.2, 0.75, 0.02, -0.01), 245.0 * np.exp(move))


def test_certain_flat_answers_the_reference_price():
    assert two_step_point(245.0, 1.0, 0.75, 0.02, -0.01) == 245.0


def test_missing_history_raises_so_the_caller_falls_back_to_pr_a():
    with pytest.raises(ValueError):
        two_step_point(245.0, 0.2, 0.75, np.nan, -0.01)


def history():
    days = pd.bdate_range("2026-01-05", periods=4, tz=TZ)
    rows = pd.DataFrame({"date": days, "symbol": "KBANK", "session": "morning", "p_start": 100.0, "p_end": [101.0, 99.0, 102.0, 500.0]})
    other = rows.assign(symbol="ADVANC", p_end=900.0)
    pm = rows.assign(session="afternoon", p_end=900.0)
    return pd.concat([rows, other, pm], ignore_index=True), days


def test_history_uses_only_earlier_days_of_the_same_stock_and_session():
    ds, days = history()
    up, dn = cond_r_means(ds, "KBANK", "morning", days[3])
    assert np.isclose(up, np.mean([np.log(1.01), np.log(1.02)])) and np.isclose(dn, np.log(0.99))


def test_flat_days_do_not_enter_the_conditional_means():
    ds, days = history()
    ds.loc[(ds["symbol"] == "KBANK") & (ds["session"] == "morning") & (ds["date"] == days[0]), "p_end"] = 100.0
    up, _ = cond_r_means(ds, "KBANK", "morning", days[3])
    assert np.isclose(up, np.log(1.02))
