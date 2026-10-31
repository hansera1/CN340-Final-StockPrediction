"""ดึงแท่งราคาจาก yfinance (<SYMBOL>.BK) → data/raw/yfinance/<SYMBOL>_<interval>.csv

ย้อนหลังได้ (ตรวจ 29 ต.ค. 2026): 1m ~8 วันทำการ, 2m ~1 เดือน, 5m/15m/30m 60 วัน, 60m ~2 ปี
timestamp = เวลาเริ่มแท่ง; ไม่มีแท่งก่อน 10:00 และไม่มีแท่งประมูลปิด; volume ช่วงเช้าเชื่อไม่ได้ (แท่ง 10:00 มักเป็น 0)
"""
import pandas as pd

MAX_PERIOD = {"1m": "8d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d", "60m": "730d", "1d": "max"}


def ticker(symbol):
    return f"{symbol}.BK"


def fetch(symbol, interval, rules, period=None):
    """→ DataFrame[timestamp (เวลาไทยไม่มี tz), open, high, low, close, volume] — รวมแท่งที่ยังไม่จบด้วย"""
    import yfinance as yf
    df = yf.download(ticker(symbol), interval=interval, period=period or MAX_PERIOD[interval],
                     progress=False, auto_adjust=False, threads=False)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index
    idx = idx.tz_convert(rules["timezone"]).tz_localize(None) if idx.tz is not None else idx
    out = pd.DataFrame({"timestamp": idx, "open": df["Open"].to_numpy(), "high": df["High"].to_numpy(),
                        "low": df["Low"].to_numpy(), "close": df["Close"].to_numpy(),
                        "volume": df["Volume"].to_numpy()})
    return out.dropna(subset=["close"]).reset_index(drop=True)
