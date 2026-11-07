"""E19 ขั้น 0: นับวันที่มี 1m ภาคเช้าครบ + สร้างภาพ (ดัดแปลงจาก 25_1m_scalogram_demo.py ของเพื่อน ตามแผนรอบ 7 ข้อ 10)

    python experiments/27_e19_1m_scalogram.py
→ data/processed/e19_images.npz, results/e19/days.csv
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import PROCESSED_DIR, ROOT, load_rules, raw_source_dir
from forecast.data.io import load_bars
from forecast.features.scalogram_1m import image_for_day, morning_bars, pick_source


def load_1m(sym, rules, src):
    try:
        return load_bars(sym, rules, raw_source_dir(rules, src), "1m")
    except FileNotFoundError:
        return None


def source_match(st, yf, days, rules):
    """yfinance ตรงกับ Settrade แค่ไหนบนวันที่มีทั้งคู่ (สัดส่วนนาทีที่ราคาปิดเท่ากัน)"""
    rows = []
    for d in days:
        a, b = morning_bars(st, d, rules), morning_bars(yf, d, rules)
        if len(a) and len(b):
            j = a[["bar_start", "close"]].merge(b[["bar_start", "close"]], on="bar_start", suffixes=("_st", "_yf"))
            if len(j):
                rows.append((j["close_st"] == j["close_yf"]).mean())
    return float(np.mean(rows)) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=1, choices=[1, 5], help="5 = รวมแท่ง 1m ชุดเดียวกันเป็นแท่ง 5 นาที → e19_images_5m.npz")
    args = ap.parse_args()
    suffix = "" if args.minutes == 1 else f"_{args.minutes}m"
    rules = load_rules()
    rows, imgs, keys = [], [], []
    for sym in rules["symbols"]:
        st, yf = load_1m(sym, rules, "settrade"), load_1m(sym, rules, "yfinance")
        days = sorted(set(st["date"].unique()) | set(yf["date"].unique()))
        both = sorted(set(st["date"].unique()) & set(yf["date"].unique()))
        print(f"{sym}: ราคาปิด yfinance ตรง Settrade {source_match(st, yf, both, rules):.1%} ของนาที ({len(both)} วันร่วม)")
        for d in days:
            bars, name = pick_source(st, yf, d, rules)
            img = image_for_day(bars, d, rules, args.minutes) if bars is not None else None
            rows.append({"symbol": sym, "date": str(pd.Timestamp(d).date()), "source": name, "n_bars": 0 if bars is None else len(morning_bars(bars, d, rules)), "used": img is not None})
            if img is not None:
                imgs.append(img)
                keys.append(f"{sym}|{pd.Timestamp(d).date()}")
    t = pd.DataFrame(rows)
    (ROOT / "results" / "e19").mkdir(parents=True, exist_ok=True)
    t.to_csv(ROOT / "results" / "e19" / f"days{suffix}.csv", index=False, encoding="utf-8-sig")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_DIR / f"e19_images{suffix}.npz", img=np.stack(imgs), keys=np.array(keys))
    print(t.groupby(["symbol", "source"], dropna=False)["used"].agg(["size", "sum"]).to_string())
    n = t[t["used"]].groupby("symbol")["date"].nunique()
    print(f"ภาพ {len(imgs)} ภาพ · วันต่อหุ้น {n.to_dict()} · ช่วง {t[t['used']]['date'].min()} – {t[t['used']]['date'].max()}")


if __name__ == "__main__":
    main()
