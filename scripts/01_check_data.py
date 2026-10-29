"""ตรวจคุณภาพข้อมูลดิบและจำนวนตัวอย่างจริงต่อหุ้น/รอบ (หัวข้อ 03)

    python scripts/01_check_data.py [--raw-dir data/raw/_synthetic]
    → outputs/reports/data_quality_<SYMBOL>.csv

พิมพ์เวลาแท่งแรก/สุดท้ายของแต่ละภาคและแท่งหลัง 16:30 เพื่อยืนยัน bar_timestamp จากข้อมูลจริง
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, load_rules
from forecast.data.io import load_bars
from forecast.data.quality import daily_quality, summarize_quality, timestamp_profile
from forecast.labels.make_labels import make_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path)
    args = ap.parse_args()
    rules = load_rules()
    out = OUTPUT_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)

    for sym in rules["symbols"]:
        bars = load_bars(sym, rules, args.raw_dir)
        q = daily_quality(bars, rules)
        q.to_csv(out / f"data_quality_{sym}.csv", index=False)
        labels = make_labels(bars, rules)
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
