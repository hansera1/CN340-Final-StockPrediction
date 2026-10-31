"""แท่งสำหรับวันจริง = ประวัติจากไฟล์หลัก (yfinance 60m) ถึงเมื่อวาน + แท่ง 1m ของวันนี้ที่รวมเป็น 60m แบบ yfinance

แหล่ง 1m ของวันนี้ลองตามลำดับใน rules.yaml → live.sources (Settrade ก่อน, yfinance สำรอง)
แหล่งที่ไม่มีไฟล์ / ไม่มีแท่งวันนี้ / แท่งล่าสุดเก่ากว่า live.max_bar_lag_min → ข้ามไปแหล่งถัดไป
"""
import pandas as pd

from forecast.config import interval_minutes, raw_source_dir, session_times
from forecast.data.io import aggregate_bars, load_bars


def today_bars_1m(symbol, rules, date, as_of, source_dirs=None):
    """→ (แท่ง 1m วันนี้ที่จบ ณ/ก่อน as_of, ชื่อแหล่ง, ความหน่วงวินาที, เหตุผลของแหล่งที่ข้าม)"""
    max_lag = pd.Timedelta(minutes=rules["live"]["max_bar_lag_min"])
    skipped = []
    for src in rules["live"]["sources"]:
        folder = (source_dirs or {}).get(src) or raw_source_dir(rules, src)
        try:
            b = load_bars(symbol, rules, folder, "1m")
        except FileNotFoundError:
            skipped.append(f"{src}: ไม่มีไฟล์")
            continue
        b = b[(b["date"] == date) & (b["bar_end"] <= as_of)]
        if b.empty:
            skipped.append(f"{src}: ไม่มีแท่งวันนี้")
            continue
        lag = as_of - b["bar_end"].max()
        if lag > max_lag:
            skipped.append(f"{src}: แท่งล่าสุดเก่า {lag.total_seconds() / 60:.0f} นาที")
            continue
        return b, src, lag.total_seconds(), skipped
    return None, None, None, skipped


def live_bars(symbol, rules, date, as_of, raw_dir=None, source_dirs=None):
    """→ (แท่งตาม data.interval พร้อมใช้กับ assemble_dataset, ข้อมูลแหล่ง)"""
    hist = load_bars(symbol, rules, raw_dir)
    hist = hist[hist["date"] < date]
    meta = {"source": None, "lag_sec": None, "skipped": []}

    market_open = session_times(rules, "morning", date)["start"]
    if as_of <= market_open:          # รอบเช้า: ยังไม่มีแท่งของวันนี้ ใช้แค่ประวัติ
        meta["source"] = "history-only"
        return hist.reset_index(drop=True), meta

    today, src, lag, skipped = today_bars_1m(symbol, rules, date, as_of, source_dirs)
    meta.update(source=src, lag_sec=lag, skipped=skipped)
    if today is None:
        return hist.reset_index(drop=True), meta
    agg = aggregate_bars(today, interval_minutes(rules["data"]["interval"]), rules)
    agg["symbol"] = symbol
    return pd.concat([hist, agg], ignore_index=True), meta
