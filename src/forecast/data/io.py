"""โหลดแท่งรายนาทีจาก data/raw/<SYMBOL>_1min.csv แล้วจัดให้มี bar_start / bar_end ชัดเจน

รูปแบบไฟล์ดิบ: timestamp, close, volume (จำเป็น) + open, high, low, value (ถ้ามี)
"""
import pandas as pd

from forecast.config import RAW_DIR


def raw_path(symbol, raw_dir=None):
    return (raw_dir or RAW_DIR) / f"{symbol}_1min.csv"


def load_bars(symbol, rules, raw_dir=None):
    df = pd.read_csv(raw_path(symbol, raw_dir))
    ts = pd.to_datetime(df["timestamp"])
    tz = rules["timezone"]
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)

    freq = pd.Timedelta(minutes=rules["bar_minutes"])
    if rules["bar_timestamp"] == "end":
        df["bar_end"], df["bar_start"] = ts, ts - freq
    else:
        df["bar_start"], df["bar_end"] = ts, ts + freq

    df = df.drop(columns="timestamp")
    df = df.sort_values("bar_end").drop_duplicates("bar_end", keep="last")
    df["date"] = df["bar_end"].dt.normalize()
    df["symbol"] = symbol
    return df.reset_index(drop=True)


def resample_bars(bars, minutes):
    """รวมแท่งตามนิยาม OHLC และ volume ภายในวัน (ไม่ข้ามพักเที่ยง/ข้ามคืนเป็นแท่งเดียว)"""
    if minutes <= 1:
        return bars
    freq = f"{minutes}min"
    b = bars.copy()
    b["bucket"] = b["bar_start"].dt.floor(freq)
    agg = {"close": "last", "volume": "sum", "bar_end": "last", "date": "first", "symbol": "first"}
    if "open" in b:
        agg["open"] = "first"
    if "high" in b:
        agg["high"] = "max"
    if "low" in b:
        agg["low"] = "min"
    out = b.groupby("bucket", sort=True).agg(agg).reset_index()
    # แท่งรวมจะ "จบ" เมื่อแท่งย่อยสุดท้ายในกลุ่มจบ — ใช้ bar_end จริงเพื่อไม่เห็นอนาคตเกินจุดตัด
    return out.rename(columns={"bucket": "bar_start"})
