"""E18b: ĝ นอกตัวอย่างไม่รั่วข้อมูล และการเลื่อนราคาไม่แตะแถวบ่ายหรือ feature"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT

TZ = "Asia/Bangkok"


@pytest.fixture(scope="module")
def e18b():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("38_e18b_gap_shift")


def frame(e18b, n=170, seed=0):
    days = pd.bdate_range("2025-01-06", periods=n, tz=TZ)
    rng = np.random.default_rng(seed)
    parts = []
    for sym in ("KBANK", "ADVANC"):
        x = rng.normal(0, 0.01, (n, len(e18b.FX)))
        parts.append(pd.DataFrame(x, columns=e18b.FX).assign(date=days, symbol=sym, session="morning", gap=0.5 * x[:, 0] + rng.normal(0, 0.003, n)))
    return pd.concat(parts, ignore_index=True)


def test_gap_model_trains_only_before_each_row(e18b):
    gh, _ = e18b.gap_oos(frame(e18b))
    e18b.assert_no_leak(gh)
    h = gh[gh["gap_model_train_end"].notna()]
    assert len(h) and (h["gap_model_train_end"] < h["date"]).all()


def test_future_gaps_do_not_change_earlier_predictions(e18b):
    a = frame(e18b)
    cut = np.sort(a["date"].unique())[140]
    b = a.copy()
    b.loc[b["date"] >= cut, "gap"] = 9.0
    ga, gb = [e18b.gap_oos(x)[0].sort_values(e18b.KEYS).reset_index(drop=True) for x in (a, b)]
    early = ga["date"] <= cut
    assert np.allclose(ga.loc[early, "g_hat"], gb.loc[early, "g_hat"])
    assert not np.allclose(ga.loc[~early, "g_hat"], gb.loc[~early, "g_hat"])


def test_rows_before_minimum_training_are_zero(e18b):
    a = frame(e18b)
    gh, _ = e18b.gap_oos(a)
    first = np.sort(a["date"].unique())[:120]
    h = gh[gh["date"].isin(first)]
    assert (h["g_hat"] == 0).all() and h["gap_model_train_end"].isna().all()


def test_leak_check_fails_when_training_includes_the_row(e18b):
    gh, _ = e18b.gap_oos(frame(e18b))
    bad = gh.copy()
    i = bad["gap_model_train_end"].notna().idxmax()
    bad.loc[i, "gap_model_train_end"] = bad.loc[i, "date"]
    with pytest.raises(AssertionError):
        e18b.assert_no_leak(bad)


def test_asia_gap_averages_available_markets_and_holiday_is_zero(e18b):
    fx = pd.DataFrame({"fx_n225_gap": [0.01, np.nan, np.nan], "fx_kospi_gap": [0.03, 0.02, np.nan]})
    assert np.allclose(e18b.add_asia_gap(fx)["fx_asia_gap"], [0.02, 0.02, 0.0])


def test_shift_changes_only_morning_reference_and_rounds_to_tick(e18b):
    day = pd.Timestamp("2026-06-01", tz=TZ)
    ds = pd.DataFrame({"date": [day, day], "symbol": ["KBANK", "KBANK"], "session": ["morning", "afternoon"], "p_ref": [199.0, 199.0], "f_x": [1.0, 2.0]})
    gh = pd.DataFrame({"date": [day], "symbol": ["KBANK"], "session": ["morning"], "g_hat": [0.0126]})
    d = e18b.shifted(ds, gh).set_index("session")
    assert d.loc["morning", "pc"] == 199.0 and d.loc["morning", "p_ref"] == 202.0     # 201.52: ช่องราคาคิดจากระดับใหม่ (1 บาท)
    assert d.loc["afternoon", "p_ref"] == 199.0 and d.loc["afternoon", "g_hat"] == 0.0
    assert d.loc["morning", "f_x"] == 1.0


def test_conditional_means_use_only_earlier_days(e18b):
    days = pd.bdate_range("2026-01-05", periods=4, tz=TZ)
    am = pd.DataFrame({"date": days, "symbol": "KBANK", "session": "morning", "p_ref": 100.0, "p_end": [101.0, 99.0, 102.0, 500.0]})
    m = e18b.cond_means(am)
    assert np.isnan(m["z_up"].iloc[0]) and np.isclose(m["z_up"].iloc[1], np.log(1.01))
    assert np.isclose(m["z_up"].iloc[3], np.mean([np.log(1.01), np.log(1.02)]))


def test_mean_point_formula(e18b):
    assert np.isclose(e18b.mean_point(100.0, 0.2, 0.5, 0.02, -0.01), 100.0 * np.exp(0.8 * 0.5 * 0.02 + 0.8 * 0.5 * -0.01))


def test_rmse_bootstrap_recomputes_from_days_not_row_differences(e18b):
    ea, eb = np.array([3.0, 0.0]), np.array([0.0, 0.0])
    m, lo, hi = e18b.boot_rmse(ea, eb, np.array(["a", "b"]), 0.95, 500)
    assert np.isclose(m, np.sqrt(4.5))
    assert lo == 0.0 and hi == 3.0


def test_verdict_needs_every_criterion(e18b):
    rows = []
    for name in e18b.MODELS:
        for g in ("เช้า", "KBANK เช้า", "ADVANC เช้า"):
            rows.append({"model": name, "group": g, "d_rmse_pra_hi": -0.1, "d_rmse_pra_lo": -0.2, "d_mae_pra_lo": -0.1})
    t = pd.DataFrame(rows)
    t.loc[(t["model"] == "H2b-rule") & (t["group"] == "เช้า"), "d_rmse_pra_hi"] = 0.1
    t.loc[(t["model"] == "H2b-mean") & (t["group"] == "KBANK เช้า"), "d_rmse_pra_lo"] = 0.1
    cov = pd.DataFrame([{"model": m, "group": "เช้า", "cover50": 0.5, "cover80": 0.8} for m in ["PR-A"] + e18b.MODELS])
    v = e18b.verdict(t, cov).set_index("model")["pass"]
    assert v["H2b-0"] and not v["H2b-rule"] and not v["H2b-mean"]
