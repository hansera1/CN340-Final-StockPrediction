"""ล็อกวิธี: ฝึกโมเดลที่เลือกบนข้อมูลที่มีทั้งหมดแล้วบันทึกไว้ใช้ส่งจริง (5–6 พ.ย.)

    python scripts/04_lock_model.py [--exp E1 --variant independent] [--raw-dir ...]
    → outputs/models/<E>/<variant>/<SYMBOL>_<session>.pkl

ค่า default มาจาก rules.yaml → live; หลังล็อกห้ามเปิดค้นพารามิเตอร์ใหม่ระหว่างสัปดาห์จริง
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, load_experiments, load_rules
from forecast.evaluation.runner import fit_final, load_or_build_dataset, sequences_for


def main():
    rules = load_rules()
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=rules["live"]["experiment"])
    ap.add_argument("--variant", default=rules["live"]["variant"])
    ap.add_argument("--raw-dir", type=Path)
    args = ap.parse_args()

    v = next(v for v in load_experiments()[args.exp]["variants"] if v["name"] == args.variant)
    ds = load_or_build_dataset(rules, v.get("feature_minutes", 1), v.get("spectral", False),
                               args.raw_dir, rebuild=True)
    seqs = sequences_for(ds, rules, args.raw_dir) if v["model"] == "cnn1d" else None
    paths = fit_final(ds, v, rules, args.exp, OUTPUT_DIR / "models" / args.exp / args.variant, seqs)
    for p in paths:
        print("บันทึก", p)
    if (args.exp, args.variant) != (rules["live"]["experiment"], rules["live"]["variant"]):
        print("\nอย่าลืมแก้ live.experiment / live.variant ใน configs/rules.yaml ให้ตรงกับโมเดลนี้")


if __name__ == "__main__":
    main()
