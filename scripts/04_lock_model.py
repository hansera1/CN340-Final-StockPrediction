"""ล็อกวิธี: ฝึกโมเดลที่เลือกบนข้อมูลที่มีทั้งหมดแล้วบันทึกไว้ใช้ส่งจริง

    python scripts/04_lock_model.py                      # ทุกโมเดลใน rules.yaml → live (ทิศทาง, ราคา 2 ตัว, เงา)
    python scripts/04_lock_model.py --spec E9/lrx        # เฉพาะตัวที่ระบุ
    → outputs/models/<E>/<variant>/<SYMBOL>_<session>.pkl

refit ตามกติกา (ศ. 6 พ.ย.) ใช้คำสั่งเดียวกัน — ค่า hyperparameter / features มาจาก experiments.yaml ไม่เปลี่ยน
หลังล็อกห้ามเปิดค้นพารามิเตอร์ใหม่ระหว่างสัปดาห์จริง
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, load_experiments, load_rules
from forecast.evaluation.runner import fit_final, load_or_build_dataset, variant_interval

LIVE_KEYS = ["direction", "price_eq", "price_up", "shadow"]


def main():
    rules = load_rules()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", help="เช่น E9/lrx (default = ทุกตัวใน live)")
    ap.add_argument("--raw-dir", type=Path)
    args = ap.parse_args()
    specs = args.spec or list(dict.fromkeys(rules["live"][k] for k in LIVE_KEYS))

    ds = load_or_build_dataset(rules, raw_dir=args.raw_dir, rebuild=True)
    print(f"ข้อมูลฝึก: {ds['date'].min().date()} – {ds.loc[ds['up'].notna(), 'date'].max().date()} ({len(ds)} แถว)")
    for spec in specs:
        code, name = spec.split("/")
        exp = load_experiments()[code]
        if exp.get("pilot"):
            raise SystemExit(f"{code} เป็น pilot (ข้อมูลน้อย) — ไม่ควรล็อกเป็นโมเดลส่งจริง")
        v = next(v for v in exp["variants"] if v["name"] == name)
        if variant_interval(rules, exp, v) != rules["data"]["interval"]:
            raise SystemExit(f"{spec} ไม่ได้ใช้ {rules['data']['interval']} ซึ่งเป็นแท่งที่วันจริงใช้สร้าง features")
        for p in fit_final(ds, v, rules, code, OUTPUT_DIR / "models" / code / name):
            print("บันทึก", p)


if __name__ == "__main__":
    main()
