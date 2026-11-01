"""E13: ฝึกชั้น 2 ตอนเย็นสำหรับรอบบ่ายของวันทำการถัดไป (results/e13_preregistration.md)

    python scripts/09_train_stage2.py [--date 2026-11-02]    # default = วันทำการถัดไป (รันใน 00_collect_daily.py ให้เอง)
→ outputs/models/stage2/<วันที่>.json (β, mean, sd, จำนวนวัน/แถวฝึก หรือเหตุผลที่ใช้ p₁)
"""
import argparse
from datetime import datetime

import _setup  # noqa: F401

from forecast.config import load_rules
from forecast.live.daily_check import next_trading_day
from forecast.live.stage2_live import train_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="วันที่ทาย (default = วันทำการถัดไป)")
    args = ap.parse_args()
    rules = load_rules()
    if not rules["live"]["stage2"]["enabled"]:
        print("live.stage2.enabled = false → ไม่ฝึก")
        return
    date = args.date or f"{next_trading_day(datetime.now()):%Y-%m-%d}"
    info, path = train_for(rules, date)
    print(f"ชั้น 2 สำหรับรอบบ่าย {date}: วันฝึก {info['n_days']} ({info['first_day']} – {info['last_day']}), "
          f"แถวไม่ flat {info['n_nonflat']}")
    if info["model"]:
        print("  β =", dict(zip(info["features"], [round(b, 4) for b in info["model"]["beta"]])))
    else:
        print("  ", info["reason"])
    print("→", path)


if __name__ == "__main__":
    main()
