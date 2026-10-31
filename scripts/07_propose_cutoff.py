"""เสนอค่า c_เช้า / c_บ่าย จากผลวัดจริง

    python scripts/07_propose_cutoff.py [--buffer-min 5]

อ่าน outputs/logs/latency.csv (จาก 00_fetch_*.py --latency) และ outputs/logs/run_time.csv (จาก 05_predict_live.py)
    c = เวลาส่ง − (ความหน่วงสูงสุด + เวลารันนานสุด + เผื่อส่งผล)
รอบเช้าตัดก่อนตลาดไทยเปิด จึงไม่ขึ้นกับความหน่วงของแท่งวันนี้ — ใช้เฉพาะเวลารัน
ใช้ค่าสูงสุดของแหล่งแรกที่ใช้งานได้ใน live.sources และของแหล่งสำรองด้วย (ถ้าต้องสลับแหล่งต้องยังทัน)
"""
import argparse

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules
from forecast.data.latency import propose_cutoff, summarize_latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-min", type=float, default=5)
    args = ap.parse_args()
    rules = load_rules()
    logs = OUTPUT_DIR / "logs"

    lat_path, run_path = logs / "latency.csv", logs / "run_time.csv"
    if not lat_path.exists():
        raise SystemExit("ยังไม่มี latency.csv — รัน 00_fetch_yfinance.py / 00_fetch_settrade.py --latency 30 ช่วงตลาดเปิด")
    lat = pd.read_csv(lat_path, parse_dates=["bar_end", "first_seen"])
    lat["part"] = lat["bar_end"].dt.hour.map(lambda h: "morning" if h < 13 else "afternoon")
    print("ความหน่วงต่อแหล่ง / ภาค:")
    worst = 0.0
    for (src, part), g in lat.groupby(["source", "part"]):
        s = summarize_latency(g.reset_index(drop=True))
        print(f"  {src:9s} {part:9s} {s}")
        worst = max(worst, s.get("delay_max_sec") or 0)

    if run_path.exists():
        runs = pd.read_csv(run_path)
        runs = runs[runs["fetched"]]
        run_sec = runs.groupby("session")["run_sec"].max().to_dict()
    else:
        run_sec = {}
        print("ยังไม่มี run_time.csv — ใช้เวลารัน 120 วินาทีชั่วคราว (ซ้อม 05_predict_live.py เพื่อวัดจริง)")
    print("\nเวลารันนานสุด (ดึงข้อมูล + ทำนาย):", run_sec or "-")

    for sess, cfg in rules["sessions"].items():
        delay = worst if sess == "afternoon" else 0.0
        c = propose_cutoff(cfg["submit_deadline"], delay, run_sec.get(sess, 120), args.buffer_min)
        print(f"  {sess:9s}: ส่ง {cfg['submit_deadline']} → เสนอ c = {c}  (ตอนนี้ {cfg['cutoff']}; "
              f"หน่วง {delay:.0f} วิ + รัน {run_sec.get(sess, 120):.0f} วิ + เผื่อ {args.buffer_min:g} นาที)")
    print("\nแก้ใน configs/rules.yaml → sessions.<รอบ>.cutoff เมื่อยืนยันแล้ว")


if __name__ == "__main__":
    main()
