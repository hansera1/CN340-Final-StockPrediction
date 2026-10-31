"""ดึงแท่งราคาจาก yfinance → data/raw/yfinance/<SYMBOL>_<interval>.csv

    python scripts/00_fetch_yfinance.py                         # ชุด backtest: 1d + 60m (~2 ปี) + 5m/15m/30m (60 วัน) + 1m
    python scripts/00_fetch_yfinance.py --interval 1m           # วันจริง: แท่ง 1m ล่าสุด (สำรองของ Settrade)
    python scripts/00_fetch_yfinance.py --latency 30            # วัดความหน่วง 30 นาที (ช่วงตลาดเปิด)

ต่อท้ายไฟล์เดิม ไม่แก้แถวที่มีอยู่ และตัดแท่งที่ยังไม่จบ ณ เวลาที่ดึง — รันซ้ำได้เพื่อสะสมข้อมูล
"""
import argparse

import _setup  # noqa: F401
import pandas as pd

from forecast.config import INTERVAL_MIN, OUTPUT_DIR, load_rules, raw_source_dir
from forecast.data.io import drop_unfinished, merge_into_raw, raw_path
from forecast.data.latency import measure_latency, summarize_latency
from forecast.data.yfinance_src import MAX_PERIOD, fetch

SOURCE = "yfinance"
BACKTEST_SET = ["1d", "60m", "30m", "15m", "5m", "1m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", nargs="+", choices=list(MAX_PERIOD), default=BACKTEST_SET)
    ap.add_argument("--period", help="เช่น 5d (default = ย้อนได้ไกลสุดของ interval นั้น)")
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--latency", type=int, metavar="MINUTES", help="วัดความหน่วงกี่นาที")
    args = ap.parse_args()
    rules = load_rules()
    tz = rules["timezone"]
    symbols = args.symbols or rules["symbols"]
    out_dir = raw_source_dir(rules, SOURCE)

    if args.latency:
        log = OUTPUT_DIR / "logs" / "latency.csv"
        lat = measure_latency(lambda: fetch(symbols[0], "1m", rules, period="1d").tail(5), rules, args.latency,
                              log, SOURCE)
        print(summarize_latency(lat), "→", log)
        print("เสนอค่า c: python scripts/07_propose_cutoff.py")
        return

    for iv in args.interval:
        for sym in symbols:
            now = pd.Timestamp.now(tz=tz).tz_localize(None)
            new = drop_unfinished(fetch(sym, iv, rules, args.period), rules, now, INTERVAL_MIN[iv])
            if new.empty:
                print(f"  {sym} {iv}: ไม่ได้ข้อมูล")
                continue
            added, diff = merge_into_raw(raw_path(sym, rules, out_dir, iv), new)
            days = new["timestamp"].dt.normalize().nunique()
            print(f"  {sym} {iv}: {len(new)} แท่ง {days} วัน ({new['timestamp'].min():%Y-%m-%d} → "
                  f"{new['timestamp'].max():%Y-%m-%d %H:%M}), เพิ่มใหม่ {added}"
                  + (f", ⚠️ ค่าไม่ตรงของเดิม {diff} (เก็บของเดิม)" if diff else ""))


if __name__ == "__main__":
    main()
