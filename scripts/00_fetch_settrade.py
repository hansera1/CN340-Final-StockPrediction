"""ดึงแท่งรายนาทีจาก Settrade Open API → data/raw/<SYMBOL>_1min.csv

ตั้งรหัสใน .env ก่อน (คัดลอกจาก .env.example) — ห้าม commit / ห้ามวางรหัสในโค้ดหรือแชต

    python scripts/00_fetch_settrade.py --probe                      # ดู response ดิบ 1 วัน (ครั้งแรกควรรันก่อน)
    python scripts/00_fetch_settrade.py --start 2024-01-01           # ดึงย้อนหลังถึงวันนี้ ต่อท้ายไฟล์เดิม
    python scripts/00_fetch_settrade.py --start today                # วันจริง: ดึงข้อมูลวันนี้ก่อนรัน 05
    python scripts/00_fetch_settrade.py --latency 30                 # วัดความหน่วง 30 นาที (ช่วงตลาดเปิด)
    python scripts/00_fetch_settrade.py --interval 60m               # เก็บแท่งยาวกว่า → data/raw/<SYMBOL>_60m.csv

ข้อจำกัดของ API (ตรวจ 29 ต.ค. 2026): 1m ย้อนได้ ~4 วันทำการ, 5m ~7 วัน, 15m ~1 เดือน, 60m ~3 เดือน, 1d ~3 ปี
→ ต้องรันทุกวันเพื่อสะสมรายนาที ข้อมูลที่หลุดช่วงไปแล้วดึงคืนไม่ได้
"""
import argparse
import time

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, RAW_DIR, load_rules
from forecast.data.io import raw_path
from forecast.data.settrade import (INTERVAL_MIN, candles_to_frame, connect, drop_unfinished,
                                    fetch_day, fetch_latest, measure_latency, merge_into_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYY-MM-DD หรือ today")
    ap.add_argument("--end", default="today")
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--probe", action="store_true", help="พิมพ์ response ดิบของวันทำการล่าสุด ไม่เขียนไฟล์")
    ap.add_argument("--latency", type=int, metavar="MINUTES", help="วัดความหน่วงกี่นาที")
    ap.add_argument("--interval", choices=list(INTERVAL_MIN), default="1m",
                    help="นอกจาก 1m: ดึงทั้งหมดที่ API ยังเก็บไว้ในคำขอเดียว ไม่ต้องใช้ --start")
    ap.add_argument("--sleep", type=float, default=0.3, help="พักระหว่างคำขอ (วินาที)")
    args = ap.parse_args()
    rules = load_rules()
    tz = rules["timezone"]
    symbols = args.symbols or rules["symbols"]
    today = pd.Timestamp.now(tz=tz).tz_localize(None).normalize()
    day = lambda s: today if s == "today" else pd.Timestamp(s)
    market = connect()

    if args.probe:
        d = today - pd.offsets.BDay(1)
        res = market.get_candlestick(symbol=symbols[0], interval="1m", limit=1000,
                                     start=d.strftime("%Y-%m-%dT00:00"),
                                     end=(d + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00"))
        print("keys:", sorted(res))
        df = candles_to_frame(res, rules)
        print(f"{symbols[0]} {d.date()}: {len(df)} แท่ง\n{df.head(3)}\n...\n{df.tail(3)}")
        return

    if args.latency:
        log = OUTPUT_DIR / "logs" / "latency.csv"
        log.parent.mkdir(parents=True, exist_ok=True)
        lat = measure_latency(market, symbols[0], rules, args.latency, log)
        print(f"\nความหน่วงสูงสุด {lat['lag_sec'].max():.0f} วิ, คำขอนานสุด {lat['request_sec'].max():.2f} วิ"
              f" → บันทึก {log}")
        print("ตั้ง cutoff = เวลาส่ง − (ความหน่วงสูงสุด + เวลารันโมเดล + เผื่อส่ง 5 นาที)")
        return

    if args.interval != "1m":
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        now = pd.Timestamp.now(tz=tz).tz_localize(None)
        for sym in symbols:
            new = drop_unfinished(fetch_latest(market, sym, args.interval, rules), rules, now,
                                  INTERVAL_MIN[args.interval])
            added, diff = merge_into_raw(RAW_DIR / f"{sym}_{args.interval}.csv", new)
            print(f"  {sym} {args.interval}: {len(new)} แท่ง ({new['timestamp'].min()} → "
                  f"{new['timestamp'].max()}), เพิ่มใหม่ {added}" + (f", ⚠️ ค่าไม่ตรงของเดิม {diff}" if diff else ""))
        return

    if not args.start:
        ap.error("ต้องระบุ --start (หรือ --probe / --latency)")
    dates = pd.bdate_range(day(args.start), day(args.end))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        frames, empty = [], 0
        for i, d in enumerate(dates, 1):
            df = fetch_day(market, sym, d, rules)
            if df.empty:
                empty += 1                       # วันหยุดหรือไม่มีข้อมูลย้อนหลังถึง
            frames.append(df)
            print(f"\r  {sym} {d.date()} ({i}/{len(dates)})", end="", flush=True)
            time.sleep(args.sleep)
        new = pd.concat(frames, ignore_index=True)
        new = drop_unfinished(new, rules, pd.Timestamp.now(tz=tz).tz_localize(None))
        added, diff = merge_into_raw(raw_path(sym), new)
        print(f"\r  {sym}: ได้ {len(new)} แท่ง, เพิ่มใหม่ {added}, วันที่ไม่มีข้อมูล {empty}/{len(dates)}"
              + (f", ⚠️ ค่าไม่ตรงของเดิม {diff} จุด (เก็บของเดิมไว้)" if diff else ""))
        if len(new):
            print(f"      ข้อมูลเก่าสุดที่ได้: {new['timestamp'].min()}")
    print("\nถัดไป: python scripts/01_check_data.py")


if __name__ == "__main__":
    main()
