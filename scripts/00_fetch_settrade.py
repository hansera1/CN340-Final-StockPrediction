"""ดึงแท่งราคาจาก Settrade Open API → data/raw/settrade/<SYMBOL>_<interval>.csv

ตั้งรหัสใน .env ก่อน (คัดลอกจาก .env.example) — ห้าม commit / ห้ามวางรหัสในโค้ดหรือแชต

    python scripts/00_fetch_settrade.py --probe                      # ดู response ดิบ 1 วัน (ครั้งแรกควรรันก่อน)
    python scripts/00_fetch_settrade.py --start 2026-10-26           # 1m ทีละวัน ต่อท้ายไฟล์เดิม
    python scripts/00_fetch_settrade.py --start today                # วันจริง: ข้อมูลวันนี้
    python scripts/00_fetch_settrade.py --interval 60m               # interval อื่น: ทั้งหมดที่ API ยังเก็บไว้
    python scripts/00_fetch_settrade.py --latency 30                 # วัดความหน่วง 30 นาที (ช่วงตลาดเปิด)

ข้อจำกัดของ API (ตรวจ 29 ต.ค. 2026): 1m ย้อนได้ ~4 วันทำการ, 5m ~7 วัน, 15m ~1 เดือน, 60m ~3 เดือน, 1d ~3 ปี
→ ต้องรันทุกวันเพื่อสะสมรายนาที ข้อมูลที่หลุดช่วงไปแล้วดึงคืนไม่ได้
"""
import argparse
import time

import _setup  # noqa: F401
import pandas as pd

from forecast.config import INTERVAL_MIN, OUTPUT_DIR, load_rules, raw_source_dir
from forecast.data.io import drop_unfinished, merge_into_raw, raw_path
from forecast.data.latency import measure_latency, summarize_latency
from forecast.data.settrade import candles_to_frame, connect, fetch_day, fetch_last, fetch_latest

SOURCE = "settrade"


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
    out_dir = raw_source_dir(rules, SOURCE)
    now = lambda: pd.Timestamp.now(tz=tz).tz_localize(None)
    today = now().normalize()
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
        lat = measure_latency(lambda: fetch_last(market, symbols[0], rules), rules, args.latency, log, SOURCE)
        print(summarize_latency(lat), "→", log)
        print("เสนอค่า c: python scripts/07_propose_cutoff.py")
        return

    if args.interval != "1m":
        for sym in symbols:
            new = drop_unfinished(fetch_latest(market, sym, args.interval, rules), rules, now(),
                                  INTERVAL_MIN[args.interval])
            added, diff = merge_into_raw(raw_path(sym, rules, out_dir, args.interval), new)
            print(f"  {sym} {args.interval}: {len(new)} แท่ง ({new['timestamp'].min()} → "
                  f"{new['timestamp'].max()}), เพิ่มใหม่ {added}" + (f", ⚠️ ค่าไม่ตรงของเดิม {diff}" if diff else ""))
        return

    if not args.start:
        ap.error("ต้องระบุ --start (หรือ --probe / --latency / --interval)")
    dates = pd.bdate_range(day(args.start), day(args.end))
    for sym in symbols:
        frames, empty = [], 0
        for i, d in enumerate(dates, 1):
            df = fetch_day(market, sym, d, rules)
            if df.empty:
                empty += 1                       # วันหยุดหรือไม่มีข้อมูลย้อนหลังถึง
            frames.append(df)
            print(f"\r  {sym} {d.date()} ({i}/{len(dates)})", end="", flush=True)
            time.sleep(args.sleep)
        new = drop_unfinished(pd.concat(frames, ignore_index=True), rules, now(), 1)
        added, diff = merge_into_raw(raw_path(sym, rules, out_dir, "1m"), new)
        print(f"\r  {sym}: ได้ {len(new)} แท่ง, เพิ่มใหม่ {added}, วันที่ไม่มีข้อมูล {empty}/{len(dates)}"
              + (f", ⚠️ ค่าไม่ตรงของเดิม {diff} จุด (เก็บของเดิมไว้)" if diff else ""))
        if len(new):
            print(f"      ข้อมูลเก่าสุดที่ได้: {new['timestamp'].min()}")


if __name__ == "__main__":
    main()
