"""ราคา ณ จุดเวลา — ใช้รายการล่าสุดที่จบแล้วเท่านั้น ไม่ใช้รายการถัดไปเติมย้อนกลับ"""
import numpy as np
import pandas as pd


def bars_until(bars, cutoff):
    """เฉพาะแท่งที่จบแล้ว ณ จุดตัด"""
    return bars[bars["bar_end"] <= cutoff]


def last_close(bars, ts):
    """ราคาปิดของแท่งล่าสุดที่จบก่อนหรือเท่ากับ ts → (ราคา, เวลาแท่ง)"""
    idx = pd.DatetimeIndex(bars["bar_end"])
    i = idx.searchsorted(ts, side="right") - 1
    if i < 0:
        return np.nan, pd.NaT
    return float(bars["close"].iat[i]), idx[i]


def price_points(day_bars):
    """แปลงแท่งเป็นจุดราคา: open ที่ bar_start และ close ที่ bar_end

    ถ้าเวลาเดียวกัน open ของแท่งถัดไปเกิดทีหลัง close ของแท่งก่อน จึงเก็บ open
    ทำให้ P(10:00) = ราคาเปิดของแท่งแรกเมื่อ timestamp เป็นเวลาจบแท่ง
    """
    closes = pd.Series(day_bars["close"].to_numpy(), index=pd.DatetimeIndex(day_bars["bar_end"]))
    parts = [closes]
    if "open" in day_bars and day_bars["open"].notna().any():
        parts.append(pd.Series(day_bars["open"].to_numpy(),
                               index=pd.DatetimeIndex(day_bars["bar_start"])).dropna())
    s = pd.concat(parts).sort_index(kind="stable")
    return s[~s.index.duplicated(keep="last")]


def price_asof(points, ts):
    """จุดราคาล่าสุดที่เวลา <= ts → (ราคา, เวลา)"""
    i = points.index.searchsorted(ts, side="right") - 1
    if i < 0:
        return np.nan, pd.NaT
    return float(points.iat[i]), points.index[i]


def price_at_or_after(points, ts, tolerance):
    """จุดราคาแรกที่เวลา >= ts ภายใน tolerance — ใช้เฉพาะสร้าง label ย้อนหลัง"""
    i = points.index.searchsorted(ts, side="left")
    if i >= len(points) or points.index[i] - ts > tolerance:
        return np.nan, pd.NaT
    return float(points.iat[i]), points.index[i]
