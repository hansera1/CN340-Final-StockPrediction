"""E27: feature ข้ามคืนใช้เฉพาะข้อมูลก่อน 08:30 และคำนวณ XD ถูกหุ้นถูกวัน"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT

TZ = "Asia/Bangkok"
DAYS = pd.bdate_range("2026-03-16", periods=6, tz=TZ)


@pytest.fixture(scope="module")
def e27():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("54_e27_gap_analysis")


def table(seed=0, n=30):
    idx = pd.bdate_range("2026-03-02", periods=n)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return pd.DataFrame({"Open": c * 1.002, "Close": c, "Adj Close": c}, index=idx)


def tables():
    return {t: table(i) for i, t in enumerate(["Z74.SI", "D05.SI", "^BKX", "TLT", "^TNX"])}


def test_singapore_open_is_available_before_the_cutoff(e27):
    t = tables()
    f = e27.build_new(DAYS, t).set_index("date")
    d = DAYS[3]
    prev = t["Z74.SI"].loc[t["Z74.SI"].index < d.tz_localize(None).normalize(), "Close"].iloc[-1]
    assert np.isclose(f.loc[d, "fx_singtel_gap"], np.log(t["Z74.SI"].loc[d.tz_localize(None).normalize(), "Open"] / prev))


def test_us_close_of_the_same_day_is_not_used(e27):
    t = tables()
    d = DAYS[3]
    a = e27.build_new(DAYS[:4], t).set_index("date")
    changed = {k: v.copy() for k, v in t.items()}
    day = d.tz_localize(None).normalize()
    for k in ("^BKX", "TLT", "^TNX"):
        changed[k].loc[changed[k].index >= day, ["Close", "Adj Close"]] *= 3.0
    b = e27.build_new(DAYS[:4], changed).set_index("date")
    assert np.allclose(a.loc[d, ["fx_bkx", "fx_tlt", "fx_tnx"]].astype(float), b.loc[d, ["fx_bkx", "fx_tlt", "fx_tnx"]].astype(float))


def test_close_of_the_same_day_in_singapore_is_not_used(e27):
    t = tables()
    d = DAYS[3]
    a = e27.build_new(DAYS[:4], t).set_index("date")
    changed = {k: v.copy() for k, v in t.items()}
    day = d.tz_localize(None).normalize()
    changed["Z74.SI"].loc[day, "Close"] *= 3.0
    b = e27.build_new(DAYS[:4], changed).set_index("date")
    assert np.isclose(a.loc[d, "fx_singtel_gap"], b.loc[d, "fx_singtel_gap"])


def test_xd_applies_only_to_its_symbol_and_day(e27):
    am = pd.DataFrame({"date": ["2026-04-21", "2026-04-21", "2026-04-22"], "symbol": ["KBANK", "ADVANC", "KBANK"], "p_ref": [200.0, 300.0, 200.0]})
    cal = pd.DataFrame({"date": ["2026-04-21", "2026-04-21"], "symbol": ["KBANK", "ALL"], "event": ["XD", "holiday"], "amount": [12.0, np.nan]})
    f, amt = e27.xd_feature(am, cal)
    assert np.isclose(f[0], np.log(188 / 200)) and f[1] == 0.0 and f[2] == 0.0 and amt.tolist() == [12.0, 0.0, 0.0]
