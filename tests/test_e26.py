"""E26 ตามแบบ V8: feature ใหม่ใช้เฉพาะข้อมูลก่อนจุดตัด"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT, load_rules, session_times

RULES = load_rules()
TZ = RULES["timezone"]


@pytest.fixture(scope="module")
def e26():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("53_e26_features")


def daily(n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-05", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": c * np.exp(rng.normal(0, 0.003, n)), "close": c}, index=idx)


def bars(n=8):
    days = pd.bdate_range("2026-03-02", periods=n, tz=TZ)
    ends = [d + pd.Timedelta(h) for d in days for h in ("11:00:00", "12:00:00", "12:30:00", "15:00:00", "16:00:00", "16:30:00")]
    rng = np.random.default_rng(0)
    return pd.DataFrame({"bar_end": ends, "date": [e.normalize() for e in ends], "close": 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, len(ends))))})


def test_last_before_excludes_the_same_day(e26):
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-08"]))
    got = e26.last_before(s, pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-09"]))
    assert np.isnan(got[0]) and got[1:].tolist() == [1.0, 2.0, 3.0]


def test_daily_features_use_only_earlier_days(e26):
    d = daily()
    days = d.index[[15, 25]]
    a = e26.daily_features(d, days)
    changed = d.copy()
    changed.loc[changed.index >= days[0], ["open", "close"]] *= 5.0
    b = e26.daily_features(changed, days)
    assert np.allclose(a.loc[days[0]], b.loc[days[0]])


def test_overnight_and_intraday_split_adds_to_the_daily_return(e26):
    d = daily()
    f = e26.daily_features(d, d.index[[10]]).iloc[0]
    prev = d.index[9]
    assert np.isclose(f["fo_on_1"] + f["fo_id_1"], np.log(d.loc[prev, "close"] / d.loc[d.index[8], "close"]))


def test_market_feature_is_the_previous_day_return(e26):
    d = daily()["close"]
    got = e26.market_feature(d, d.index[[10]])[0]
    assert np.isclose(got, np.log(d.iloc[9] / d.iloc[8]))


def test_rv_after_cutoff_has_no_effect(e26):
    b = bars()
    day = b["date"].unique()[6]
    rows = pd.DataFrame({"date": [day, day], "session": ["morning", "afternoon"]})
    cut_am, cut_pm = [session_times(RULES, s, day)["cutoff"] for s in ("morning", "afternoon")]
    a = e26.rv_features(b, rows, RULES)
    changed = b.copy()
    changed.loc[changed["bar_end"] > cut_pm, "close"] *= 3.0
    c = e26.rv_features(changed, rows, RULES)
    assert np.allclose(a["fv_rv60_am"].to_numpy()[1], c["fv_rv60_am"].to_numpy()[1])
    assert np.allclose(a[["fv_rv60_1", "fv_rv60_5", "fv_rv60_20"]].to_numpy(), c[["fv_rv60_1", "fv_rv60_5", "fv_rv60_20"]].to_numpy(), equal_nan=True)
    assert np.isnan(a["fv_rv60_am"].iloc[0]) and cut_am < cut_pm


def test_with_features_handles_both_formats(e26):
    extra = {"morning": ["a"], "afternoon": ["a", "b"]}
    assert e26.with_features({"features": ["x"]}, extra)["features"] == ["x", "a", "b"]
    assert e26.with_features({"features": {"morning": ["x"], "afternoon": ["y"]}}, extra)["features"] == {"morning": ["x", "a"], "afternoon": ["y", "a", "b"]}


def test_abs_r_prediction_uses_only_earlier_days(e26):
    days = pd.bdate_range("2025-01-06", periods=170, tz=TZ).to_numpy()
    rng = np.random.default_rng(0)
    g = pd.DataFrame({"date": days, "symbol": "KBANK", "session": "morning", "p_start": 100.0, "p_end": 100 * np.exp(rng.normal(0, 0.01, 170)), "x": rng.normal(size=170)})
    blocks = [days[i:i + 5] for i in range(120, 170, 5)]
    a = e26.predict_abs_r(g, ["x"], blocks)
    changed = g.copy()
    changed.loc[140:, "p_end"] = 900.0
    b = e26.predict_abs_r(changed, ["x"], blocks)
    assert np.allclose(a["abs_r"].to_numpy()[:140], b["abs_r"].to_numpy()[:140], equal_nan=True)
