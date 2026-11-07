"""E23c: ภาพ 5m ไม่รั่วข้อมูลหลังจุดตัด และแถวที่แท่งไม่ครบถูกตัดออก"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT, load_rules, session_times

RULES = load_rules()
TZ = RULES["timezone"]
DAYS = pd.bdate_range("2026-11-02", periods=4, tz=TZ)


@pytest.fixture(scope="module")
def e23c():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("50_e23c_5m")


def bars():
    ends = []
    for d in DAYS:
        ends += list(pd.date_range(d + pd.Timedelta("10:05:00"), d + pd.Timedelta("12:30:00"), freq="5min"))
        ends += list(pd.date_range(d + pd.Timedelta("14:05:00"), d + pd.Timedelta("16:30:00"), freq="5min"))
    rng = np.random.default_rng(0)
    b = pd.DataFrame({"bar_end": ends, "close": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(ends)))), "volume": rng.integers(100, 1000, len(ends)).astype(float)})
    b["date"] = b["bar_end"].dt.normalize()
    return b


def rows(day, session="afternoon"):
    return pd.DataFrame({"date": [day], "symbol": ["KBANK"], "session": [session]})


@pytest.mark.parametrize("session", ["morning", "afternoon"])
def test_bars_after_cutoff_do_not_change_the_image(e23c, session):
    b = bars()
    day = DAYS[3]
    cutoff = session_times(RULES, session, day)["cutoff"]
    changed = b.copy()
    after = changed["bar_end"] > cutoff
    assert after.any()
    changed.loc[after, "close"] *= 3.0
    changed.loc[after, "volume"] *= 50.0
    _, a = e23c.images_5m(rows(day, session), {"KBANK": b}, RULES)
    _, c = e23c.images_5m(rows(day, session), {"KBANK": changed}, RULES)
    assert a.shape == (1, 3, 56, 56) and np.array_equal(a, c)


def test_rows_with_too_few_bars_are_dropped(e23c):
    g, x = e23c.images_5m(pd.concat([rows(DAYS[0]), rows(DAYS[3])], ignore_index=True), {"KBANK": bars()}, RULES)
    assert g["date"].tolist() == [DAYS[3]] and len(x) == 1


def test_period_labels(e23c):
    got = e23c.period(["2026-10-12", "2026-10-14", "2026-10-29", "2026-10-30", "2026-11-06"]).tolist()
    assert got == ["walk-forward", "holdout", "holdout", "ข้อมูลใหม่", "ข้อมูลใหม่"]
