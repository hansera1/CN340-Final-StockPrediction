"""โหลดแท่งราคาจาก data/raw/<source>/<SYMBOL>_<interval>.csv แล้วจัดให้มี bar_start / bar_end ชัดเจน

รูปแบบไฟล์ดิบ: timestamp, close, volume (จำเป็น) + open, high, low, value (ถ้ามี)

bar_end ของแท่งที่คร่อมเวลาปิดภาคถูกตัดที่เวลาปิดภาค (เช่นแท่ง 60m ที่เริ่ม 12:00 จบ 12:30 ไม่ใช่ 13:00)
→ ที่จุดตัด 12:15 แท่งนี้ยัง "ไม่จบ" จึงไม่ถูกใช้ทั้งใน backtest และวันจริง
"""
import pandas as pd

from forecast.config import interval_minutes, raw_source_dir

OHLCV = ["open", "high", "low", "close", "volume"]


def raw_path(symbol, rules, raw_dir=None, interval=None):
    interval = interval or rules["data"]["interval"]
    return (raw_dir or raw_source_dir(rules)) / f"{symbol}_{interval}.csv"


def session_ends(rules, days):
    """เวลาปิดของแต่ละภาค ต่อวัน → list ของ Series (index ตรงกับ days)"""
    out = []
    for cfg in rules["sessions"].values():
        h, m = cfg["end_time"].split(":")
        out.append(days + pd.Timedelta(hours=int(h), minutes=int(m)))
    return out


def clip_to_session_end(bar_start, bar_end, rules):
    """แท่งที่เริ่มก่อนเวลาปิดภาคแต่จบหลังจากนั้น → จบที่เวลาปิดภาค"""
    bar_end = bar_end.copy()
    for end in session_ends(rules, bar_start.dt.normalize()):
        cross = (bar_start < end) & (bar_end > end)
        bar_end[cross] = end[cross]
    return bar_end


def frame_from_raw(df, rules, interval):
    """DataFrame ที่มีคอลัมน์ timestamp (เวลาไทย) → แท่งพร้อม bar_start / bar_end / date"""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    tz = rules["timezone"]
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)

    freq = pd.Timedelta(minutes=interval_minutes(interval))
    if rules["bar_timestamp"] == "end":
        df["bar_end"], df["bar_start"] = ts, ts - freq
    else:
        df["bar_start"], df["bar_end"] = ts, ts + freq
    df["bar_end"] = clip_to_session_end(df["bar_start"], df["bar_end"], rules)

    df = df.drop(columns="timestamp")
    df = df.sort_values("bar_end").drop_duplicates("bar_start", keep="last")
    df["date"] = df["bar_start"].dt.normalize()
    return df.reset_index(drop=True)


def load_bars(symbol, rules, raw_dir=None, interval=None):
    interval = interval or rules["data"]["interval"]
    df = frame_from_raw(pd.read_csv(raw_path(symbol, rules, raw_dir, interval)), rules, interval)
    df["symbol"] = symbol
    return df


def aggregate_bars(bars, minutes, rules):
    """รวมแท่งให้เป็นแท่ง `minutes` นาทีแบบเดียวกับ yfinance (ตรวจกับข้อมูลจริง 26–29 ต.ค. 2026)

    - กลุ่มตามนาฬิกา (60m → 10:00, 11:00, ...) ไม่ข้ามวัน
    - ตัดแท่งก่อนเวลาเริ่มภาคเช้า (ช่วงเปิดแบบสุ่ม 09:55–10:00) และแท่งประมูลปิดหลัง 16:30 ทิ้ง
      เพราะ yfinance ไม่มีสองช่วงนี้; ส่วนการซื้อขาย 13:5x ก่อนเปิดภาคบ่ายเก็บเป็นแท่ง 13:00 เหมือน yfinance
    - bar_end = เวลาจบตามนาฬิกาของกลุ่ม (ตัดที่เวลาปิดภาค) ไม่ใช่เวลาจบของแท่งย่อยสุดท้าย
      → กลุ่มที่ยังไม่ครบ ณ จุดตัดมี bar_end > จุดตัด จึงไม่ถูกใช้
    """
    day = bars["bar_start"].dt.normalize()
    first, last = rules["sessions"]["morning"]["start_time"], rules["sessions"]["afternoon"]["end_time"]
    at = lambda hhmm: day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    b = bars[(bars["bar_start"] >= at(first)) & (bars["bar_start"] < at(last))].copy()
    if minutes <= 1 or b.empty:
        return b.reset_index(drop=True)

    b["bucket"] = b["bar_start"].dt.floor(f"{minutes}min")
    agg = {"close": "last", "volume": "sum", "date": "first"}
    for col, how in {"open": "first", "high": "max", "low": "min", "symbol": "first"}.items():
        if col in b:
            agg[col] = how
    out = b.groupby("bucket", sort=True).agg(agg).reset_index().rename(columns={"bucket": "bar_start"})
    out["bar_end"] = clip_to_session_end(out["bar_start"], out["bar_start"] + pd.Timedelta(minutes=minutes), rules)
    return out


def drop_unfinished(df, rules, now, minutes):
    """ตัดแท่งที่ยังไม่จบ ณ เวลาที่ดึง (คอลัมน์ timestamp เวลาไทยไม่มี tz) — ไม่ให้ค่าครึ่งแท่งถูกบันทึกถาวร"""
    freq = pd.Timedelta(minutes=minutes)
    end = df["timestamp"] + (freq if rules["bar_timestamp"] == "start" else pd.Timedelta(0))
    return df[end <= now]


def merge_into_raw(path, new):
    """ต่อท้ายไฟล์เดิม ไม่แก้แถวที่มีอยู่แล้ว (data/raw ห้ามแก้ย้อนหลัง) → (จำนวนแถวใหม่, แถวที่ค่าไม่ตรงของเดิม)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["timestamp"])
        both = old.merge(new, on="timestamp", suffixes=("", "_new"))
        diff = int(sum((both[c] != both[f"{c}_new"]).sum() for c in ("close", "volume") if c in old))
        added = new[~new["timestamp"].isin(old["timestamp"])]
        out = pd.concat([old, added], ignore_index=True)
    else:
        added, diff, out = new, 0, new
    out.sort_values("timestamp").to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return len(added), diff
