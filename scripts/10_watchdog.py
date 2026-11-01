"""งาน 9 ข้อ 4: ถึงเส้นตาย − 10 นาทียังไม่มีไฟล์ส่ง → แจ้งเตือนกลุ่ม (Task Scheduler 09:20 / 12:20 — scripts/setup_tasks.ps1)

    python scripts/10_watchdog.py morning
    python scripts/10_watchdog.py afternoon
วันหยุด (calendar.csv) / เสาร์–อาทิตย์ → ไม่ตรวจ
"""
import argparse
import sys
from datetime import datetime

import _setup  # noqa: F401

from forecast.config import ROOT, load_rules, session_times
from forecast.live.daily_check import is_trading_day
from forecast.live.notify import notify

SESSION_TH = {"morning": "เช้า", "afternoon": "บ่าย"}


def submission_file(rules, date, session):
    sub = rules["submission"]
    return ROOT / sub["dir"] / sub["filename"].format(date=date, session=session, session_th=SESSION_TH[session])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", choices=["morning", "afternoon"])
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    rules = load_rules()
    if not is_trading_day(args.date):
        print(f"{args.date} ไม่ใช่วันทำการ — ไม่ตรวจ")
        return 0
    path = submission_file(rules, args.date, args.session)
    deadline = session_times(rules, args.session, args.date)["deadline"]
    if path.exists():
        print(f"มีไฟล์ส่งแล้ว: {path.name}")
        return 0
    notify("missing", f"{args.date} {SESSION_TH[args.session]}: ยังไม่มีไฟล์ส่ง — เส้นตาย {deadline:%H:%M}",
           [f"ไม่พบ {path.relative_to(ROOT)}", "รัน: python scripts/05_predict_live.py " + args.session,
            "ไม่ทัน → แผนสำรองใน docs/live_runbook.md (ร่างรอบเช้า / กรอกมือ)", f"ตรวจโดย {_setup.RUN['host']}"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
