"""E18 V8: ทุกค่าที่ใช้มีเวลาพร้อมใช้ ≤ 08:30 ของวันไทย D — ข้อมูลหลังเวลานั้นต้องไม่มีผล"""
import numpy as np
import pandas as pd

from forecast.features.overnight import FEATURES, build_features, close_series, fx_series, gap, local_ts

TZ = "Asia/Bangkok"


def table(start="2026-03-02", n=30, base=100.0, seed=0):
    idx = pd.bdate_range(start, periods=n)
    c = base * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return pd.DataFrame({"Open": c * 1.001, "Close": c, "Adj Close": c}, index=idx)


def tables():
    return {t: table(seed=i) for i, t in enumerate(["^GSPC", "THD", "^N225", "^KS11", "THB=X", "^SET50"])}


DAYS = pd.bdate_range("2026-03-16", periods=8, tz=TZ)


def test_us_close_uses_dst_time_in_bangkok():
    assert local_ts(["2026-01-15"], "16:00", "America/New_York")[0].strftime("%m-%d %H:%M") == "01-16 04:00"
    assert local_ts(["2026-07-15"], "16:00", "America/New_York")[0].strftime("%m-%d %H:%M") == "07-16 03:00"


def test_us_session_of_day_d_is_not_used_for_day_d():
    """ราคาปิดสหรัฐของวัน D ออกหลัง 08:30 ไทยวัน D → ห้ามถูกใช้ (ใช้ของ D−1)"""
    t = tables()
    base = build_features(DAYS, t)
    d = DAYS[3]
    moved = {k: v.copy() for k, v in t.items()}
    moved["^GSPC"].loc[pd.Timestamp(d.date()), "Close"] *= 1.5
    after = build_features(DAYS, moved)
    row = lambda x: x[x["date"] == d]["fx_spx"].iat[0]
    assert row(base) == row(after)
    nxt = lambda x: x[x["date"] == DAYS[4]]["fx_spx"].iat[0]
    assert nxt(base) != nxt(after)


def test_asian_close_of_day_d_is_not_used_but_open_is():
    t = tables()
    base = build_features(DAYS, t)
    d = DAYS[3]
    a, b = {k: v.copy() for k, v in t.items()}, {k: v.copy() for k, v in t.items()}
    a["^N225"].loc[pd.Timestamp(d.date()), "Close"] *= 1.5
    b["^N225"].loc[pd.Timestamp(d.date()), "Open"] *= 1.5
    row = lambda x: x[x["date"] == d]["fx_n225_gap"].iat[0]
    assert row(build_features(DAYS, a)) == row(base)
    assert row(build_features(DAYS, b)) != row(base)


def test_future_rows_never_change_past_features():
    t = tables()
    cut = pd.Timestamp(DAYS[4].date())
    base = build_features(DAYS[:5], t)
    moved = {k: v.copy() for k, v in t.items()}
    for v in moved.values():
        v.loc[v.index > cut, ["Open", "Close", "Adj Close"]] *= 3
    pd.testing.assert_frame_equal(base[FEATURES], build_features(DAYS[:5], moved)[FEATURES])


def test_holiday_abroad_gives_missing_gap_and_monday_uses_friday_us_close():
    t = tables()
    d = DAYS[3]
    t["^N225"] = t["^N225"].drop(index=pd.Timestamp(d.date()))
    r = build_features(DAYS, t)
    assert np.isnan(r[r["date"] == d]["fx_n225_gap"].iat[0])
    mon = pd.Timestamp("2026-03-23", tz=TZ)
    assert mon in DAYS
    s = close_series(t["^GSPC"], "^GSPC")
    assert s[s.index <= mon.normalize() + pd.Timedelta("08:30:00")].index[-1].strftime("%a") == "Sat"   # ราคาปิดศุกร์ = เช้าเสาร์ไทย


def test_fx_daily_available_next_utc_day_and_hourly_at_bar_end():
    d = table()
    s = fx_series(d)
    assert s.index[0].strftime("%m-%d %H:%M") == "03-03 07:00"
    h = pd.DataFrame({"Close": [30.0]}, index=pd.to_datetime(["2026-03-20 00:00:00+00:00"]))
    hs = fx_series(d, h)
    assert pd.Timestamp("2026-03-20 08:00", tz=TZ) in hs.index


def test_gap_requires_open_before_cut():
    d = table()
    day = pd.Timestamp(d.index[5], tz=TZ)
    assert np.isfinite(gap(d, "^N225", day, day.normalize() + pd.Timedelta("08:30:00")))
    assert np.isnan(gap(d, "^N225", day, day.normalize() + pd.Timedelta("06:30:00")))
