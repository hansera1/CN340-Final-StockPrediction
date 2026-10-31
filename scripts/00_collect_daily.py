"""สะสมข้อมูลทุกวันหลังตลาดปิด (D8) — รันหลัง 17:00 ทุกวันทำการ

    python scripts/00_collect_daily.py            # ก่อน 17:00 จะไม่รัน (กันได้แท่งไม่ครบวัน) เว้นแต่ใส่ --force

- Settrade 1m: ย้อน 5 วันทำการ (API เก็บ ~4 วัน) → ถ้าพลาดไปวันสองวันยังเก็บทัน
- Settrade 30m / 1d, yfinance 1d / 60m / 30m / 15m / 5m / 1m: ทั้งหมดที่ยังดึงได้
ทุกไฟล์ต่อท้ายแบบไม่ซ้ำ (merge_into_raw) — แถวที่มีอยู่แล้วไม่ถูกแก้
แหล่งไหนล้มจะข้ามไปแหล่งถัดไป แล้วสรุปท้ายว่าอะไรล้ม
"""
import argparse
import subprocess
import sys
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import load_rules

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="รันแม้ยังไม่ถึง 17:00")
    args = ap.parse_args()
    rules = load_rules()
    now = pd.Timestamp.now(tz=rules["timezone"]).tz_localize(None)
    if now.hour < 17 and not args.force:
        sys.exit(f"ตอนนี้ {now:%H:%M} — รอหลัง 17:00 (หรือใส่ --force)")

    start = (now.normalize() - pd.offsets.BDay(5)).strftime("%Y-%m-%d")
    jobs = [
        ("settrade 1m", ["00_fetch_settrade.py", "--start", start]),
        ("settrade 30m", ["00_fetch_settrade.py", "--interval", "30m"]),
        ("settrade 1d", ["00_fetch_settrade.py", "--interval", "1d"]),
        ("yfinance", ["00_fetch_yfinance.py"]),
    ]
    failed = []
    for name, cmd in jobs:
        print(f"\n## {name}")
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]])
        if r.returncode:
            failed.append(name)
    print("\nสรุป:", "สำเร็จทั้งหมด" if not failed else f"ล้ม: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
