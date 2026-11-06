"""ล็อกวิธี: ฝึกโมเดลที่เลือกบนข้อมูลที่มีทั้งหมดแล้วบันทึกไว้ใช้ส่งจริง

    python scripts/04_lock_model.py                      # ทุกโมเดลใน rules.yaml → live (ทิศทาง, ราคา 2 ตัว, เงา)
    python scripts/04_lock_model.py --spec E9/lrx        # เฉพาะตัวที่ระบุ
    python scripts/04_lock_model.py --atc --target up_notup --spec E12/fd_lr   # เป้าหมายขึ้น/ไม่ขึ้น → outputs/models/R4ATC_UN/
    python scripts/04_lock_model.py --atc                # ชุดสำรอง label 16:30 = ATC → outputs/models/R4ATC/<E>/<variant>/
                                                         # (ใช้เมื่ออาจารย์ตอบว่า 16:30 = ATC: rules.yaml → label.price_1630
                                                         #  = closing_auction และ live.model_dir = R4ATC)
    → outputs/models/<E>/<variant>/<SYMBOL>_<session>.pkl

refit ตามกติกา (ศ. 6 พ.ย.) ใช้คำสั่งเดียวกัน — ค่า hyperparameter / features มาจาก experiments.yaml ไม่เปลี่ยน
หลังล็อกห้ามเปิดค้นพารามิเตอร์ใหม่ระหว่างสัปดาห์จริง
"""
import argparse
import copy
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, load_experiments, load_rules
from forecast.evaluation.runner import fit_final, load_or_build_dataset, variant_interval
from forecast.live.slots import UN_SUFFIX, lock_specs


def main():
    rules = load_rules()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", help="เช่น E9/lrx (default = ทุกตัวใน live)")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--atc", action="store_true", help="ฝึกด้วย label 16:30 = ราคาปิดทางการ (ATC)")
    ap.add_argument("--target", choices=["up_down", "up_notup"], default="up_down",
                    help="เป้าหมายของโมเดลทิศทาง · up_notup ต้องระบุ --spec และบันทึกโฟลเดอร์ <model_dir>_UN")
    args = ap.parse_args()
    if args.target == "up_notup" and not args.spec:
        ap.error("--target up_notup ใช้กับ --spec ของโมเดลทิศทางเท่านั้น")
    root = OUTPUT_DIR / "models" / (rules["live"].get("model_dir") or "")
    if args.atc:
        rules = copy.deepcopy(rules)
        rules["label"]["price_1630"] = "closing_auction"
        root = OUTPUT_DIR / "models" / "R4ATC"
    if args.target == "up_notup":
        root = root.with_name(root.name + UN_SUFFIX)
    print(f"label: P(c) เช้า = {rules['label'].get('p_ref_morning', 'last_bar_close')}, "
          f"16:30 = {rules['label']['price_1630']} → {root}")
    specs = args.spec or lock_specs(rules)

    ds = load_or_build_dataset(rules, raw_dir=args.raw_dir, rebuild=True)
    print(f"ข้อมูลฝึก: {ds['date'].min().date()} – {ds.loc[ds['up'].notna(), 'date'].max().date()} ({len(ds)} แถว)")
    for spec in specs:
        code, name = spec.split("/")
        exp = load_experiments()[code]
        if exp.get("pilot"):
            raise SystemExit(f"{code} เป็น pilot (ข้อมูลน้อย) — ไม่ควรล็อกเป็นโมเดลส่งจริง")
        v = next(v for v in exp["variants"] if v["name"] == name)
        if args.target == "up_notup":
            v = {**v, "params": {**v.get("params", {}), "target": "up_notup"}}
        if variant_interval(rules, exp, v) != rules["data"]["interval"]:
            raise SystemExit(f"{spec} ไม่ได้ใช้ {rules['data']['interval']} ซึ่งเป็นแท่งที่วันจริงใช้สร้าง features")
        for p in fit_final(ds, v, rules, code, root / code / name):
            print("บันทึก", p)


if __name__ == "__main__":
    main()
