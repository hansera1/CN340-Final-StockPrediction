"""สร้างตาราง dataset (labels + features) แล้วเก็บไว้ที่ data/processed/

    python scripts/02_build_dataset.py [--interval 60m] [--raw-dir ...] [--feature-minutes 5] [--spectral]
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import load_rules
from forecast.evaluation.runner import load_or_build_dataset
from forecast.features.dataset import feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--interval", help="default = rules.yaml → data.interval")
    ap.add_argument("--feature-minutes", type=int)
    ap.add_argument("--spectral", action="store_true")
    args = ap.parse_args()

    ds = load_or_build_dataset(load_rules(), args.feature_minutes, args.spectral, args.raw_dir, rebuild=True,
                               interval=args.interval)
    fc = feature_cols(ds)
    print(f"แถว: {len(ds):,}  features: {len(fc)}  ช่วงวันที่: {ds['date'].min().date()} – {ds['date'].max().date()}")
    print(ds.groupby(["symbol", "session"])["up"].agg(n="count", up_rate="mean").round(3))
    missing = ds[fc].isna().mean().sort_values(ascending=False).head(10)
    print("\nfeatures ที่หายมากสุด (สัดส่วน):\n" + missing.round(3).to_string())


if __name__ == "__main__":
    main()
