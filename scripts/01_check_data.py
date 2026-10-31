"""ตรวจคุณภาพข้อมูลดิบและจำนวนตัวอย่างจริงต่อหุ้น/รอบ (หัวข้อ 03)

    python scripts/01_check_data.py                               # ไฟล์หลักตาม rules.yaml → data
    python scripts/01_check_data.py --source settrade --interval 1m
    → outputs/reports/data_quality_<SYMBOL>.csv

พิมพ์เวลาแท่งแรก/สุดท้ายของแต่ละภาคและแท่งหลัง 16:30 เพื่อยืนยัน bar_timestamp จากข้อมูลจริง
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, load_rules, raw_source_dir
from forecast.data.io import load_bars
from forecast.data.quality import daily_quality, summarize_quality, timestamp_profile
from forecast.labels.make_labels import labels_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--source", help="โฟลเดอร์ใต้ data/raw (default = data.source)")
    ap.add_argument("--interval", help="default = data.interval")
    args = ap.parse_args()
    rules = load_rules()
    interval = args.interval or rules["data"]["interval"]
    raw_dir = args.raw_dir or (raw_source_dir(rules, args.source) if args.source else None)
    print(f"ตรวจ: {raw_dir or raw_source_dir(rules)}  interval {interval}")
    out = OUTPUT_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)

    for sym in rules["symbols"]:
        bars = load_bars(sym, rules, raw_dir, interval)
        q = daily_quality(bars, rules)
        q.to_csv(out / f"data_quality_{sym}_{interval}.csv", index=False)
        labels = labels_for(bars, rules, interval)
        print(f"\n=== {sym} ===")
        for k, v in summarize_quality(q).items():
            print(f"  {k:28s} {v}")
        print("  --- timestamp (ตามไฟล์ดิบ) ---")
        for k, v in timestamp_profile(bars, rules).items():
            print(f"  {k:28s} {v}")
        for sess, g in labels.groupby("session"):
            print(f"  label ใช้ได้ ({sess}){'':14s} {g['up'].notna().sum()} / {len(g)} วัน"
                  f"  (flat {int((g['direction'] == 0).sum())})")
    print(f"\nรายงานรายวัน: {out}")


if __name__ == "__main__":
    main()
