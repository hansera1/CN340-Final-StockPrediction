"""E18: feature ตลาดต่างประเทศของรอบเช้า — กันข้อมูลรั่วด้วยเวลาจริง ไม่ใช่วันที่ (results/e18_preregistration.md)

หน้าต่างข้ามคืนของวันไทย D = (16:30 วันทำการไทยก่อนหน้า, 08:30 วัน D]
ทุกค่าที่ใช้มีเวลาพร้อมใช้ ≤ 08:30 วัน D เสมอ · แท่งรายวันแปลงเวลาด้วย zoneinfo จึงคิด DST ให้เอง
"""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BKK = ZoneInfo("Asia/Bangkok")
FEATURES = ["fx_spx", "fx_thd_prem", "fx_n225_gap", "fx_kospi_gap", "fx_usdthb"]
CLOSE_TIME = {"^GSPC": ("16:00", "America/New_York"), "THD": ("16:00", "America/New_York"), "^N225": ("15:30", "Asia/Tokyo"),
              "^KS11": ("15:30", "Asia/Seoul"), "^SET50": ("16:40", "Asia/Bangkok")}
OPEN_TIME = {"^N225": ("09:00", "Asia/Tokyo"), "^KS11": ("09:00", "Asia/Seoul")}


def local_ts(dates, hhmm, tz):
    """วันที่ (ไม่มี tz) + เวลาท้องถิ่นของตลาด → เวลาไทย"""
    d = pd.to_datetime(dates)
    t = pd.DatetimeIndex(d.normalize() + pd.Timedelta(f"{hhmm}:00"))
    return t.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert(BKK)


def close_series(df, ticker, col="Close"):
    """DataFrame รายวัน (index วันที่) → Series ค่าปิดตามเวลาปิดจริงเป็นเวลาไทย"""
    x = df[col].dropna()
    hhmm, tz = CLOSE_TIME[ticker]
    return pd.Series(x.to_numpy(float), index=local_ts(x.index, hhmm, tz)).sort_index()


def fx_series(daily, hourly=None):
    """THB=X → Series ตามเวลาพร้อมใช้: รายวัน = 00:00 UTC ของวันถัดไป · 60m = เวลาจบแท่ง (index เวลา UTC)"""
    x = daily["Close"].dropna()
    idx = pd.DatetimeIndex(pd.to_datetime(x.index).normalize() + pd.Timedelta(days=1)).tz_localize("UTC").tz_convert(BKK)
    s = pd.Series(x.to_numpy(float), index=idx)
    if hourly is not None and len(hourly):
        h = hourly["Close"].dropna()
        i = pd.DatetimeIndex(pd.to_datetime(h.index, utc=True)) + pd.Timedelta(hours=1)
        hs = pd.Series(h.to_numpy(float), index=i.tz_convert(BKK))
        s = pd.concat([s[s.index < hs.index.min()], hs])
    return s.sort_index()


def asof(s, t):
    """ค่าล่าสุดที่เวลา ≤ t · ไม่มี = NaN"""
    i = s.index.searchsorted(t, side="right") - 1
    return float(s.iloc[i]) if i >= 0 else np.nan


def window_change(s, a, b):
    return np.log(asof(s, b)) - np.log(asof(s, a))


def gap(df, ticker, day, cut):
    """ln(Open ของวัน D) − ln(Close ล่าสุดก่อน D) · Open ต้องพร้อมใช้ก่อน cut · ไม่มีตลาดวัน D = NaN"""
    hhmm, tz = OPEN_TIME[ticker]
    d = pd.Timestamp(day.date())
    if d not in df.index or not np.isfinite(df.loc[d, "Open"]):
        return np.nan
    if local_ts([d], hhmm, tz)[0] > cut:
        return np.nan
    prev = df.loc[df.index < d, "Close"].dropna()
    return np.log(df.loc[d, "Open"]) - np.log(prev.iloc[-1]) if len(prev) else np.nan


def build_features(days, tables, hourly=None):
    """days: วันทำการไทยเรียงตามเวลา (DatetimeIndex มี tz เวลาไทย) · tables: {ticker: DataFrame รายวัน} → DataFrame ต่อวันไทย"""
    spx, set50 = close_series(tables["^GSPC"], "^GSPC"), close_series(tables["^SET50"], "^SET50")
    thd = close_series(tables["THD"], "THD", "Adj Close")
    thb = fx_series(tables["THB=X"], hourly)
    rows = []
    for i in range(1, len(days)):
        d, prev = days[i], days[i - 1]
        a, b = prev.normalize() + pd.Timedelta("16:30:00"), d.normalize() + pd.Timedelta("08:30:00")
        fx = window_change(thb, a, b)
        rows.append({"date": d, "fx_spx": window_change(spx, a, b),
                     "fx_thd_prem": window_change(thd, a, b) - window_change(set50, a, b) + fx,
                     "fx_n225_gap": gap(tables["^N225"], "^N225", d, b), "fx_kospi_gap": gap(tables["^KS11"], "^KS11", d, b), "fx_usdthb": fx})
    return pd.DataFrame(rows)
