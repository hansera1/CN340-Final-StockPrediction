"""E13: feature 5m ของภาคเช้าวันนี้ ณ จุดตัดรอบบ่าย (สเปกล็อกใน results/e13_preregistration.md)

ใช้แท่ง 5m ภาคเช้าที่เริ่ม ≥ 10:00 และจบ ≤ จุดตัด · ราคาอย่างเดียว · P_cut = ราคาปิดแท่งล่าสุดที่จบ ≤ จุดตัด
  f5_r_last15      log(P_cut / P_(cut−15)) · P_(cut−15) = ราคาปิดแท่งล่าสุดที่จบ ≤ จุดตัด − 15 นาที (ไม่มี → open แท่งแรก)
  f5_close_vs_twap log(P_cut / ค่าเฉลี่ยราคาปิดแท่ง 5m ตั้งแต่ 10:00 ถึงจุดตัด)
  f5_range_pos     (P_cut − L) / (H − L) จาก high / low แท่ง 5m ภาคเช้า (H = L → 0.5)
backtest = yfinance 5m · วันจริง = Settrade 1m รวมเป็น 5m ด้วย aggregate_bars(bars, 5, rules) (แบบ yfinance)
"""
import numpy as np
import pandas as pd

from forecast.config import session_times

F5 = ["f5_r_last15", "f5_close_vs_twap", "f5_range_pos"]


def f5_features(day_bars, date, rules):
    """day_bars: แท่ง 5m ของวันนั้น (มี bar_start / bar_end) → dict ของ F5 (ไม่มีแท่ง → NaN)"""
    t = session_times(rules, "afternoon", date)
    start = session_times(rules, "morning", date)["start"]
    b = day_bars[(day_bars["bar_start"] >= start) & (day_bars["bar_end"] <= t["cutoff"])].sort_values("bar_end")
    if b.empty:
        return dict.fromkeys(F5, np.nan)
    p_cut = float(b["close"].iat[-1])
    before = b[b["bar_end"] <= t["cutoff"] - pd.Timedelta(minutes=15)]
    p_15 = float(before["close"].iat[-1]) if len(before) else float(b["open"].iat[0])
    hi, lo = float(b["high"].max()), float(b["low"].min())
    return {"f5_r_last15": np.log(p_cut / p_15),
            "f5_close_vs_twap": np.log(p_cut / b["close"].mean()),
            "f5_range_pos": (p_cut - lo) / (hi - lo) if hi > lo else 0.5}


def f5_table(bars5, rules, dates=None):
    """แท่ง 5m ของหุ้นหนึ่ง → DataFrame[date, F5...] ต่อวัน (dates = None → ทุกวันที่มีแท่ง)"""
    rows = []
    groups = dict(tuple(bars5.groupby("date", sort=True)))
    for date in (dates if dates is not None else groups):
        day = groups.get(date)
        rows.append({"date": date, **(f5_features(day, date, rules) if day is not None else dict.fromkeys(F5, np.nan))})
    return pd.DataFrame(rows, columns=["date", *F5])
