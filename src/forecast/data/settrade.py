"""ดึงแท่งรายนาทีจาก Settrade Open API (settrade_v2) → data/raw/<SYMBOL>_1min.csv

รหัสเข้าถึงอ่านจาก environment variable หรือไฟล์ .env ที่ root โปรเจกต์ (อยู่ใน .gitignore) เท่านั้น
    SETTRADE_APP_ID, SETTRADE_APP_SECRET, SETTRADE_APP_CODE, SETTRADE_BROKER_ID
"""
import os
import sys
import time

import pandas as pd

from forecast.config import ROOT

ENV_KEYS = ["SETTRADE_APP_ID", "SETTRADE_APP_SECRET", "SETTRADE_APP_CODE", "SETTRADE_BROKER_ID"]
PRICE_COLS = ["open", "high", "low", "close", "volume", "value"]


def load_credentials(env_file=ROOT / ".env"):
    """.env มาก่อน (ค่าที่กรอกในโปรเจกต์) แล้วค่อยเติมจาก env var — ไม่พิมพ์ค่าออกมา"""
    creds = dict.fromkeys(ENV_KEYS)
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in creds and v:
                creds[k] = v
    for k in ENV_KEYS:
        creds[k] = creds[k] or os.environ.get(k)
    missing = [k for k, v in creds.items() if not v]
    if missing:
        print(f"ยังไม่ได้ตั้งค่า: {', '.join(missing)} — คัดลอก .env.example เป็น .env แล้วกรอก")
        sys.exit(1)
    return creds


def connect():
    from settrade_v2 import Investor
    c = load_credentials()
    investor = Investor(app_id=c["SETTRADE_APP_ID"], app_secret=c["SETTRADE_APP_SECRET"],
                        app_code=c["SETTRADE_APP_CODE"], broker_id=c["SETTRADE_BROKER_ID"],
                        is_auto_queue=False)
    return investor.MarketData()


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


def fetch_latest(market, symbol, interval, rules, limit=1000):
    """แท่งล่าสุดเท่าที่ API ยังเก็บไว้ (ไม่ระบุช่วงวัน) — ใช้กับ interval ยาวกว่า 1 นาที"""
    return candles_to_frame(market.get_candlestick(symbol=symbol, interval=interval, limit=limit,
                                                   normalized=False), rules)


INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "60m": 60, "1d": 1440}


def drop_unfinished(df, rules, now, minutes=None):
    """ตัดแท่งที่ยังไม่จบ ณ เวลาที่ดึง — ไม่ให้ค่าครึ่งแท่งถูกบันทึกถาวร"""
    freq = pd.Timedelta(minutes=minutes or rules["bar_minutes"])
    end = df["timestamp"] + (freq if rules["bar_timestamp"] == "start" else pd.Timedelta(0))
    return df[end <= now]


def merge_into_raw(path, new):
    """ต่อท้ายไฟล์เดิม ไม่แก้แถวที่มีอยู่แล้ว (data/raw ห้ามแก้ย้อนหลัง) → (จำนวนแถวใหม่, แถวที่ค่าไม่ตรงของเดิม)"""
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


def measure_latency(market, symbol, rules, minutes, log_path, poll_sec=60):
    """ดึงแท่งล่าสุดทุก poll_sec วินาที บันทึก: เวลาเครื่อง, timestamp แท่งล่าสุด, ความหน่วง, เวลาคำขอ"""
    tz = rules["timezone"]
    rows = []
    for _ in range(minutes):
        t0 = pd.Timestamp.now(tz=tz)
        res = market.get_candlestick(symbol=symbol, interval="1m", limit=3, normalized=False)
        t1 = pd.Timestamp.now(tz=tz)
        last = candles_to_frame(res, rules)["timestamp"].max().tz_localize(tz)
        row = {"requested_at": t0, "received_at": t1, "last_bar_timestamp": last,
               "lag_sec": (t1 - last).total_seconds(), "request_sec": (t1 - t0).total_seconds()}
        rows.append(row)
        print(f"  {t1:%H:%M:%S}  แท่งล่าสุด {last:%H:%M}  ห่าง {row['lag_sec']:.0f} วิ  คำขอ {row['request_sec']:.2f} วิ")
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        time.sleep(max(0, poll_sec - (pd.Timestamp.now(tz=tz) - t0).total_seconds()))
    return pd.DataFrame(rows)
