"""E19: ภาพ scalogram จากแท่ง 1m ภาคเช้าของวันนั้น — 1 ภาพต่อ (หุ้น, วัน) สำหรับทายรอบบ่าย

ใช้เฉพาะแท่งที่ bar_start ≥ 10:00 และ bar_end ≤ 12:30 · เติมตารางเวลา 150 นาที: นาทีไม่มีซื้อขาย = ราคาเดิม, volume 0
3 ช่อง: r, |r|, log volume → Morlet CWT (ω₀ = 6, 32 scale, s = 2 ถึง n/2 แท่ง ตาม Nyquist) → log1p → 0–1 → 224×224
"""
import numpy as np
import pandas as pd

from forecast.config import session_times
from forecast.features.images import _resize, _unit, morlet_cwt

GRID_MIN = 150
MIN_BARS = 120
S_MIN = 2.0


def morning_bars(bars, date, rules):
    """แท่ง 1m ภาคเช้าของ date → DataFrame (ว่างถ้าไม่มี) — แท่งหลัง 12:30 ไม่ถูกใช้"""
    t = session_times(rules, "morning", date)
    b = bars[(bars["date"] == date) & (bars["bar_start"] >= t["start"]) & (bars["bar_end"] <= t["end"])]
    return b.sort_values("bar_start").reset_index(drop=True)


def to_grid(b, date, rules):
    t = session_times(rules, "morning", date)
    grid = pd.date_range(t["start"], t["end"], freq="1min", inclusive="left")
    s = b.set_index("bar_start")
    close = s["close"].reindex(grid).ffill().bfill().to_numpy(float)
    volume = s["volume"].reindex(grid).fillna(0.0).to_numpy(float)
    return close, volume


def image_from_series(close, volume):
    c = np.log(close)
    r = np.diff(c, prepend=c[0]) * 100.0
    v = np.log1p(volume)
    v = (v - v.mean()) / (v.std() + 1e-8)
    chans = [_unit(np.log1p(morlet_cwt((x - x.mean()) / (x.std() + 1e-8), n_scales=32, w0=6.0, s_min=S_MIN))) for x in (r, np.abs(r), v)]
    return np.clip(_resize(np.stack(chans).astype(np.float32)), 0, 1)


def image_for_day(bars, date, rules):
    """→ ภาพ uint8 (3, 224, 224) หรือ None ถ้าแท่งจริงไม่ถึง MIN_BARS"""
    b = morning_bars(bars, date, rules)
    if len(b) < MIN_BARS:
        return None
    return np.round(image_from_series(*to_grid(b, date, rules)) * 255).astype(np.uint8)


def pick_source(settrade, yfinance, date, rules):
    """Settrade ก่อน · yfinance เมื่อ Settrade ไม่มีวันนั้น → (แท่ง, ชื่อแหล่ง) หรือ (None, None)"""
    for name, bars in (("settrade", settrade), ("yfinance", yfinance)):
        if bars is not None and len(morning_bars(bars, date, rules)) >= MIN_BARS:
            return bars, name
    return None, None
