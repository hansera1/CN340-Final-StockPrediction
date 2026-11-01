"""สะสมข้อมูลทุกวันหลังตลาดปิด (D8) — รันหลัง 17:00 ทุกวันทำการ (Task Scheduler 17:30 — scripts/setup_tasks.ps1)

    python scripts/00_collect_daily.py            # ก่อน 17:00 จะไม่รัน (กันได้แท่งไม่ครบวัน) เว้นแต่ใส่ --force
    python scripts/00_collect_daily.py --force --target-date 2026-11-09   # เช้าวันจริงถ้างานเย็นไม่สำเร็จ (05 เรียกให้เอง)

- Settrade 1m: ย้อน 5 วันทำการ (API เก็บ ~4 วัน) → ถ้าพลาดไปวันสองวันยังเก็บทัน
- Settrade 30m / 1d, yfinance 1d / 60m / 30m / 15m / 5m / 1m: ทั้งหมดที่ยังดึงได้
ทุกไฟล์ต่อท้ายแบบไม่ซ้ำ (merge_into_raw) — แถวที่มีอยู่แล้วไม่ถูกแก้
แหล่งไหนล้มจะข้ามไปแหล่งถัดไป แล้วสรุปท้ายว่าอะไรล้ม
แล้ว → ร่างรอบเช้าของวันทำการถัดไป (05_predict_live.py morning --draft — รอบที่ 5 ข้อ A5)
     → ฝึก E13 ชั้น 2 สำหรับรอบบ่ายของวันทำการถัดไป (09_train_stage2.py)
     → outputs/logs/evening_status_<วันทำการถัดไป>.json (งาน 9 ข้อ 3) + แจ้งเตือนกลุ่ม
วันหยุดตลาด (calendar.csv) / เสาร์–อาทิตย์ → ไม่รัน เว้นแต่ใส่ --force
"""
import argparse
import subprocess
import sys
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import load_rules
from forecast.live.daily_check import is_trading_day, next_trading_day
from forecast.live.evening import write_status
from forecast.live.notify import notify

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="รันแม้ยังไม่ถึง 17:00 หรือเป็นวันหยุด")
    ap.add_argument("--target-date", help="วันทำการที่เตรียมร่างรอบเช้า / ชั้น 2 ให้ (default = วันทำการถัดไป)")
    args = ap.parse_args()
    rules = load_rules()
    now = pd.Timestamp.now(tz=rules["timezone"]).tz_localize(None)
    if not args.force:
        if not is_trading_day(now):
            sys.exit(f"วันนี้ ({now:%a %d/%m}) ไม่ใช่วันทำการ — ไม่รัน (ใส่ --force เพื่อรันเอง)")
        if now.hour < 17:
            sys.exit(f"ตอนนี้ {now:%H:%M} — รอหลัง 17:00 (หรือใส่ --force)")
    target = pd.Timestamp(args.target_date) if args.target_date else next_trading_day(now)

    start = (now.normalize() - pd.offsets.BDay(5)).strftime("%Y-%m-%d")
    jobs = [
        ("settrade 1m", ["00_fetch_settrade.py", "--start", start]),
        ("settrade 30m", ["00_fetch_settrade.py", "--interval", "30m"]),
        ("settrade 1d", ["00_fetch_settrade.py", "--interval", "1d"]),
        ("yfinance", ["00_fetch_yfinance.py"]),
    ]
    ok = {}
    for name, cmd in jobs:
        print(f"\n## {name}", flush=True)
        ok[name] = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]]).returncode == 0
    # ร่างและชั้น 2 ลองเสมอ (มี fallback ของข้อมูลในตัว) — สถานะแยกอยู่ในไฟล์สถานะ
    print(f"\n## ร่างรอบเช้าของ {target:%Y-%m-%d}", flush=True)
    subprocess.run([sys.executable, str(HERE / "05_predict_live.py"), "morning", "--draft", "--date", f"{target:%Y-%m-%d}"])
    print(f"\n## E13 ชั้น 2 สำหรับรอบบ่าย {target:%Y-%m-%d}", flush=True)
    subprocess.run([sys.executable, str(HERE / "09_train_stage2.py"), "--date", f"{target:%Y-%m-%d}"])

    st, path = write_status(target, ok, {"run_by": _setup.RUN["host"], "commit": _setup.RUN["commit"]})
    bad = [k for k, v in ok.items() if not v] + [k for k in ("draft", "stage2") if not st[k]["ok"]]
    print(f"\nสรุป ({path.name}):", "สำเร็จทั้งหมด" if st["ok"] else f"ล้ม: {', '.join(bad)}")
    notify("evening" if st["ok"] else "failed",
           f"งานเย็นสำหรับ {target:%d/%m}: " + ("สำเร็จ" if st["ok"] else "ไม่สำเร็จ — " + ", ".join(bad)),
           [f"ร่างรอบเช้า: {'มี' if st['draft']['ok'] else 'ไม่มี'} ({st['draft']['model_version']})",
            f"β ชั้น 2: {'มี' if st['stage2']['ok'] else 'ไม่มี'}" + (f" — {st['stage2']['reason']}" if st['stage2']['reason'] else ""),
            f"เครื่อง {_setup.RUN['host']}"])
    sys.exit(0 if st["ok"] else 1)


if __name__ == "__main__":
    main()
