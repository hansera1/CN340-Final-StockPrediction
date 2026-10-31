"""รอบที่ 3: feature สูตร CN340 (E12) ใช้เฉพาะข้อมูลก่อนจุดตัด"""
import numpy as np
import pandas as pd

from forecast.features import cn340


def _daily(n=80, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ts = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"timestamp": ts, "open": c * (1 + rng.normal(0, 0.002, n)), "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": rng.integers(1e5, 1e6, n).astype(float)})


def test_daily_features_use_only_past_days(monkeypatch, tmp_path):
    d = _daily()
    days = pd.DatetimeIndex(d["timestamp"])
    ds = pd.DataFrame({"date": days[40:].tz_localize("Asia/Bangkok"), "symbol": "KBANK", "session": "morning",
                       "cutoff_ts": days[40:].tz_localize("Asia/Bangkok") + pd.Timedelta("09:15:00")})
    monkeypatch.setattr(cn340, "load_daily", lambda sym, base=None: d)
    monkeypatch.setattr(cn340, "asof_bar_features", lambda ds, bars, prefix="fh": pd.DataFrame(index=ds.index))
    a = cn340.add_cn340_features(ds, {}, {})
    d2 = d.copy()
    d2.loc[60:, ["open", "high", "low", "close", "volume"]] *= 3     # เปลี่ยนวันที่ 60 เป็นต้นไป
    monkeypatch.setattr(cn340, "load_daily", lambda sym, base=None: d2)
    b = cn340.add_cn340_features(ds, {}, {})
    cols = [c for c in a if c.startswith("fd_")]
    upto = ds["date"].dt.tz_localize(None) <= days[60]                # วันที่ทาย ≤ วันที่ 60 เห็นได้ถึงวันที่ 59
    assert np.allclose(a.loc[upto, cols], b.loc[upto, cols], equal_nan=True)
    assert not np.allclose(a.loc[~upto, cols], b.loc[~upto, cols], equal_nan=True)
    # ret 1 วันของวันที่ทาย = ผลตอบแทนของวันทำการก่อนหน้า
    i = 50
    row = a[ds["date"].dt.tz_localize(None) == days[i]]
    assert np.isclose(row["fd_ret_1"].iat[0], d["close"].iat[i - 1] / d["close"].iat[i - 2] - 1)


def test_bar_features_stop_at_cutoff():
    ts = pd.date_range("2025-03-03 10:00", periods=200, freq="60min")
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 0.3, 200))
    bars = pd.DataFrame({"bar_start": ts, "bar_end": ts + pd.Timedelta("60min"), "close": c, "high": c + 0.2,
                         "low": c - 0.2})
    cut = ts[120] + pd.Timedelta("75min")                          # แท่ง 120 จบก่อนจุดตัด แท่ง 121 ยังไม่จบ
    ds = pd.DataFrame({"symbol": ["KBANK"], "cutoff_ts": [cut]})
    a = cn340.asof_bar_features(ds, {"KBANK": bars})
    bars2 = bars.copy()
    bars2.loc[121:, ["close", "high", "low"]] += 50
    b = cn340.asof_bar_features(ds, {"KBANK": bars2})
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.isclose(a["fh_c_sma20"].iat[0], c[120] / c[101:121].mean())
