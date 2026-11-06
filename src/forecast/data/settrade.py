"""ดึงแท่งราคาจาก Settrade Open API (settrade_v2) → data/raw/settrade/<SYMBOL>_<interval>.csv

รหัสเข้าถึงอ่านจาก environment variable หรือไฟล์ .env ที่ root โปรเจกต์ (อยู่ใน .gitignore) เท่านั้น
    SETTRADE_APP_ID, SETTRADE_APP_SECRET, SETTRADE_APP_CODE, SETTRADE_BROKER_ID
"""
import sys

import pandas as pd

from forecast.config import ROOT

ENV_KEYS = ["SETTRADE_APP_ID", "SETTRADE_APP_SECRET", "SETTRADE_APP_CODE", "SETTRADE_BROKER_ID"]
PRICE_COLS = ["open", "high", "low", "close", "volume", "value"]


def load_credentials(env_file=ROOT / ".env"):
    """.env มาก่อน (ค่าที่กรอกในโปรเจกต์) แล้วค่อยเติมจาก env var — ไม่พิมพ์ค่าออกมา"""
    from forecast.runtime import read_env
    creds = read_env(ENV_KEYS, env_file)
    missing = [k for k, v in creds.items() if not v]
    if missing:
        print(f"ยังไม่ได้ตั้งค่า: {', '.join(missing)} — คัดลอก .env.example เป็น .env แล้วกรอก")
        sys.exit(1)
    return creds


def _login():
    from settrade_v2 import Investor
    c = load_credentials()
    investor = Investor(app_id=c["SETTRADE_APP_ID"], app_secret=c["SETTRADE_APP_SECRET"],
                        app_code=c["SETTRADE_APP_CODE"], broker_id=c["SETTRADE_BROKER_ID"],
                        is_auto_queue=False)
    return investor.MarketData()


def session_lost(e):
    """session ถูกตัดเพราะมีการ login ด้วยรหัสเดียวกันจากที่อื่น (VM 2 และ 6 พ.ย.: Status[Kicked], U-102)"""
    text = str(e)
    return "Kicked" in text or "U-102" in text or "UserSession" in text


class Market:
    """MarketData ที่ login ใหม่แล้วลองซ้ำ 1 ครั้งเมื่อ session ถูกตัด — error อื่นส่งต่อตามเดิม"""

    def __init__(self, login=_login):
        self._login = login
        self._market = login()

    def get_candlestick(self, **kwargs):
        try:
            return self._market.get_candlestick(**kwargs)
        except Exception as e:
            if not session_lost(e):
                raise
            print("Settrade: session ถูกตัด (มีการ login จากที่อื่น) → login ใหม่แล้วลองซ้ำ", flush=True)
            self._market = self._login()
            return self._market.get_candlestick(**kwargs)


def connect():
    return Market()


def candles_to_frame(res, rules):
    """response ของ get_candlestick (ลิสต์ต่อคอลัมน์ + time เป็น epoch วินาที) → DataFrame"""
    if "time" not in res:
        raise ValueError(f"รูปแบบ response ไม่ตรงที่คาด (keys: {sorted(res)}) — ลอง --probe ดู")
    df = pd.DataFrame({"timestamp": pd.to_datetime(res["time"], unit="s", utc=True)})
    for c in PRICE_COLS:
        if c in res:
            df[c] = res[c]
    df["timestamp"] = df["timestamp"].dt.tz_convert(rules["timezone"]).dt.tz_localize(None)
    return df


def fetch_day(market, symbol, date, rules, limit=1000):
    """แท่ง 1 นาทีของวันเดียว (~300 แท่ง < limit ต่อคำขอ)"""
    d = pd.Timestamp(date)
    res = market.get_candlestick(symbol=symbol, interval="1m", limit=limit,
                                 start=d.strftime("%Y-%m-%dT00:00"),
                                 end=(d + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00"),
                                 normalized=False)
    df = candles_to_frame(res, rules)
    return df[df["timestamp"].dt.normalize() == d.normalize()]


def fetch_days(market, symbol, dates, rules, on_day=None):
    """1m ทีละวัน → (frames, วันที่ว่าง, [(วัน, ชนิด error)])

    error รายวันไม่หยุดวันที่เหลือ · ผู้เรียกตัดสินว่าล้มจริงไหม (failed_days)"""
    frames, empty, errors = [], 0, []
    for d in dates:
        try:
            df = fetch_day(market, symbol, d, rules)
        except Exception as e:                     # ไม่พิมพ์ข้อความเต็ม — บอกแค่ชนิด
            errors.append((pd.Timestamp(d), type(e).__name__))
            df = None
        if df is None or df.empty:
            empty += 1
        else:
            frames.append(df)
        if on_day:
            on_day(d)
    return frames, empty, errors


def failed_days(dates, errors):
    """ล้มจริง = error ที่วันล่าสุดที่ขอ หรือทุกวัน · error เฉพาะวันเก่าไม่กระทบข้อมูลใหม่จึงไม่ถือว่าล้ม"""
    bad = {d for d, _ in errors}
    return bool(errors) and (pd.Timestamp(dates[-1]) in bad or len(bad) == len(dates))


def fetch_latest(market, symbol, interval, rules, limit=1000):
    """แท่งล่าสุดเท่าที่ API ยังเก็บไว้ (ไม่ระบุช่วงวัน) — ใช้กับ interval ยาวกว่า 1 นาที"""
    return candles_to_frame(market.get_candlestick(symbol=symbol, interval=interval, limit=limit,
                                                   normalized=False), rules)


def fetch_last(market, symbol, rules, interval="1m", limit=5):
    """แท่งล่าสุดไม่กี่แท่ง — ใช้วัดความหน่วง"""
    return fetch_latest(market, symbol, interval, rules, limit)
