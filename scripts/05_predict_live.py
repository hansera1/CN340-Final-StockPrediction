"""ส่งจริง: รันที่ c_เช้า / c_บ่าย แล้วส่งผลก่อน 09:30 / 12:30

    python scripts/05_predict_live.py morning
    python scripts/05_predict_live.py afternoon
    (ทดสอบย้อนหลัง: --date 2026-10-28 --now "2026-10-28 12:15" --raw-dir data/raw/_synthetic)

→ outputs/predictions/<วันที่>_<รอบ>.csv  (ไฟล์ส่งอาจารย์)
→ outputs/logs/predictions.jsonl            (log ก่อนรู้ผล ห้ามแก้)
"""
import argparse
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules
from forecast.live.log_schema import append_log
from forecast.live.predict import predict_session

SESSION_TH = {"morning": "เช้า", "afternoon": "บ่าย"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", choices=["morning", "afternoon"])
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--now", help="จำลองเวลาปัจจุบัน (ทดสอบย้อนหลัง)")
    ap.add_argument("--raw-dir", type=Path)
    args = ap.parse_args()
    rules = load_rules()

    records = predict_session(args.session, args.date, rules, args.raw_dir, now=args.now)
    rows = []
    for r in records:
        r.submitted_at = str(datetime.now())
        append_log(OUTPUT_DIR / "logs" / "predictions.jsonl", r)
        p = r.price_quantiles
        rows.append({
            "วันที่": r.date, "รอบ": SESSION_TH[r.session], "หุ้น": r.symbol,
            "ทิศทาง": r.direction, "P(ขึ้น)": round(r.p_up, 3),
            "ราคาคาด": p["q50"], "ช่วง 50%": f"{p['q25']:.2f}–{p['q75']:.2f}",
            "ช่วง 80%": f"{p['q10']:.2f}–{p['q90']:.2f}",
            "ราคาอ้างอิง": r.reference_price, "fallback": r.fallback_reason or "",
        })

    table = pd.DataFrame(rows)
    out = OUTPUT_DIR / "predictions" / f"{args.date}_{args.session}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"\n→ {out}")
    if table["fallback"].astype(bool).any():
        print("⚠ มีหุ้นที่ใช้ baseline (fallback) — ตรวจสาเหตุในคอลัมน์ fallback")


if __name__ == "__main__":
    main()
