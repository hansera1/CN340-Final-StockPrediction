"""สร้างข้อมูลจำลอง (ไม่ใช่ข้อมูลจริง) ไว้ทดสอบว่า pipeline รันครบ ก่อนได้ไฟล์จริง

    python scripts/00_make_synthetic_data.py
    → data/raw/_synthetic/KBANK_1m.csv, ADVANC_1m.csv
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import RAW_DIR

START_PRICE = {"KBANK": 150.0, "ADVANC": 300.0}


def make_symbol(symbol, days, rng):
    minutes = list(pd.timedelta_range("10:01:00", "12:30:00", freq="1min")) + \
        list(pd.timedelta_range("14:01:00", "16:30:00", freq="1min"))
    price = START_PRICE[symbol]
    rows = []
    for d in days:
        price *= np.exp(rng.normal(0, 0.008))   # gap ข้ามคืน
        for m in minutes:
            o = price
            price *= np.exp(rng.normal(0, 0.0007))
            hi, lo = max(o, price) * (1 + abs(rng.normal(0, 2e-4))), min(o, price) * (1 - abs(rng.normal(0, 2e-4)))
            rows.append((d + m, o, hi, lo, price, int(rng.lognormal(9, 1))))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2026-10-28")
    args = ap.parse_args()

    out = RAW_DIR / "_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(args.start, args.end)
    rng = np.random.default_rng(0)
    for sym in START_PRICE:
        df = make_symbol(sym, days, rng)
        df.to_csv(out / f"{sym}_1m.csv", index=False, float_format="%.4f")
        print(f"{sym}: {len(df):,} แท่ง, {len(days)} วัน → {out / f'{sym}_1m.csv'}")


if __name__ == "__main__":
    main()
