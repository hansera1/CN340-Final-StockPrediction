"""ข้อมูลรายวันของวันทำการก่อนต้องมาครบก่อนทำนาย (NEXT_STEPS รอบที่ 4 ข้อ 3) — fallback แบบไล่ระดับ

ขั้น 1  ตรวจ settrade/<SYM>_1d.csv (F-daily, P(c) รอบเช้า) และ yfinance/<SYM>_60m.csv (features F0) ว่ามีแถวของวันทำการก่อน
ขั้น 2  ไม่มี → ดึงซ้ำ (scripts/05_predict_live.py)
ขั้น 3  ยังไม่มี → สร้างแท่งของวันนั้นจากแหล่งสำรอง ลงสำเนาข้อมูลใน outputs/live_cache/ (ไม่แก้ data/raw:
        merge_into_raw ไม่ทับแถวเดิม ถ้าเขียนแถวสำรองลงไฟล์จริง แถวจริงของ Settrade จะเข้าไม่ได้อีก)
        1d  : รวมจาก Settrade 1m (มีแท่งประมูลปิด → close = ราคาปิดทางการรวม ATC) → yfinance 1d → yfinance 1m (ไม่มี ATC)
        60m : รวมจากแท่ง 1m (Settrade → yfinance) แบบ yfinance ด้วย aggregate_bars
ขั้น 4  ยังไม่มี → ทำนายด้วยข้อมูลล่าสุดที่มี พร้อม fallback_reason (ห้ามจบโดยไม่มีไฟล์ส่ง)
วันทำการก่อน = วันจันทร์–ศุกร์ล่าสุดก่อนวันที่ทาย ที่ไม่ใช่วันหยุดตลาดใน data/external/calendar.csv
"""
import shutil

import pandas as pd

from forecast.config import EXTERNAL_DIR, RAW_DIR
from forecast.data.io import aggregate_bars, load_bars

CHECKS = [("settrade", "1d"), ("yfinance", "60m")]


def market_holidays(path=None):
    path = path or EXTERNAL_DIR / "calendar.csv"
    if not path.exists():
        return set()
    cal = pd.read_csv(path, parse_dates=["date"])
    return set(cal.loc[cal["event"] == "holiday", "date"].dt.normalize())


def prev_trading_day(date, holidays=None):
    holidays = market_holidays() if holidays is None else holidays
    d = pd.Timestamp(date).tz_localize(None).normalize() - pd.Timedelta(days=1)
    while d.dayofweek >= 5 or d in holidays:
        d -= pd.Timedelta(days=1)
    return d


def _days(path):
    if not path.exists():
        return set()
    return set(pd.to_datetime(pd.read_csv(path, usecols=["timestamp"])["timestamp"]).dt.normalize())


def missing_daily(rules, day, root=None):
    """→ {หุ้น: ["settrade 1d", ...]} ที่ยังไม่มีแถวของ day (ว่าง = ครบ)"""
    root = root or RAW_DIR
    out = {}
    for sym in rules["symbols"]:
        miss = [f"{src} {iv}" for src, iv in CHECKS if day not in _days(root / src / f"{sym}_{iv}.csv")]
        if miss:
            out[sym] = miss
    return out


def _day_1m(sym, rules, root, src, day):
    try:
        b = load_bars(sym, rules, root / src, "1m")
    except FileNotFoundError:
        return None
    b = b[b["date"].dt.tz_localize(None) == day]
    return b if len(b) else None


def _raw_rows(bars):
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in bars]
    out = bars[["bar_start", *cols]].rename(columns={"bar_start": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    return out


def _append(path, rows):
    old = pd.read_csv(path, parse_dates=["timestamp"]) if path.exists() else rows.iloc[:0]
    out = pd.concat([old, rows[[c for c in old.columns if c in rows]]], ignore_index=True)
    out.sort_values("timestamp").to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def synth_daily(sym, rules, root, day):
    """แท่ง 1d ของ day จากแหล่งสำรอง → (แถว raw, คำอธิบาย) หรือ (None, None)"""
    b = _day_1m(sym, rules, root, "settrade", day)
    if b is not None:
        b = b.sort_values("bar_start")
        row = {"timestamp": day, "open": b["open"].iat[0], "high": b["high"].max(), "low": b["low"].min(),
               "close": b["close"].iat[-1], "volume": b["volume"].sum()}
        atc = (b["bar_start"].iat[-1] - b["bar_start"].iat[-1].normalize()) >= pd.Timedelta(hours=16, minutes=30)
        return pd.DataFrame([row]), "รวมจาก Settrade 1m" + (" (close = ATC)" if atc else " (ไม่มีแท่ง ATC)")
    yf = root / "yfinance" / f"{sym}_1d.csv"
    if yf.exists():
        d = pd.read_csv(yf, parse_dates=["timestamp"])
        d = d[d["timestamp"].dt.normalize() == day]
        if len(d):
            return d.iloc[[-1]].assign(timestamp=day), "จาก yfinance 1d"
    b = _day_1m(sym, rules, root, "yfinance", day)
    if b is not None:
        b = b.sort_values("bar_start")
        row = {"timestamp": day, "open": b["open"].iat[0], "high": b["high"].max(), "low": b["low"].min(),
               "close": b["close"].iat[-1], "volume": b["volume"].sum()}
        return pd.DataFrame([row]), "รวมจาก yfinance 1m (ไม่มี ATC → close = ราคาสุดท้ายก่อน 16:30)"
    return None, None


def synth_60m(sym, rules, root, day):
    for src in rules["live"]["sources"]:
        b = _day_1m(sym, rules, root, src, day)
        if b is not None:
            agg = aggregate_bars(b, 60, rules)
            if len(agg):
                return _raw_rows(agg), f"รวมจาก {src} 1m"
    return None, None


def build_fallback_root(rules, missing, day, out_root, src_root=None):
    """ขั้น 3: สำเนา data/raw ของหุ้นเป้าหมาย + แถวสำรองของ day → {หุ้น: [คำอธิบาย]}"""
    src_root = src_root or RAW_DIR
    if out_root.exists():
        shutil.rmtree(out_root)
    for src in {s for s, _ in CHECKS} | set(rules["live"]["sources"]):
        (out_root / src).mkdir(parents=True, exist_ok=True)
        for sym in rules["symbols"]:
            for f in (src_root / src).glob(f"{sym}_*.csv"):
                shutil.copy2(f, out_root / src / f.name)
    notes = {}
    for sym, miss in missing.items():
        for item in miss:
            src, iv = item.split()
            rows, how = (synth_daily if iv == "1d" else synth_60m)(sym, rules, out_root, day)
            if rows is None:
                notes.setdefault(sym, []).append(f"{item} ของ {day:%d/%m}: สร้างจากแหล่งสำรองไม่ได้")
                continue
            _append(out_root / src / f"{sym}_{iv}.csv", rows)
            notes.setdefault(sym, []).append(f"{item} ของ {day:%d/%m}: {how}")
    return notes
