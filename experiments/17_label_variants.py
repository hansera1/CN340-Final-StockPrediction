"""NEXT_STEPS รอบที่ 4 ข้อ 1ค / ข้อ 4 — รันโมเดลที่ล็อกบน walk-forward เดิมด้วยนิยาม label ใหม่ (ไม่แตะ holdout)

    python experiments/17_label_variants.py pcoff     # ข้อ 4: P(c) รอบเช้า = ราคาปิดทางการของวันทำการก่อน
    python experiments/17_label_variants.py atc       # ข้อ 1ค: P(16:30) = ราคาปิดทางการ (ATC) + P(c) เช้าแบบใหม่

ข้อมูลตัดที่วันสุดท้ายของ dataset walk-forward เดิม (29/10/2026) → แถว วันทดสอบ และ holdout 10 วันเดียวกับรอบที่ 3
ตรวจก่อนรันว่า label / features ที่ไม่ควรเปลี่ยนไม่เปลี่ยนจริง
→ outputs/experiments/R4<VARIANT>/<variant>_predictions.csv
"""
import copy
import sys

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset, run_variant

KEYS = ["date", "symbol", "session"]
RUN = {   # โมเดลที่ล็อก (rules.yaml → live) ที่นิยามนี้กระทบ
    "pcoff": ["E9FLAT/aprice_lr", "E9FLAT/pprice_lr", "E9/baseline_ud"],
    "atc": ["E12/fd_lr", "E12/baseline_ud", "E9FLAT/aprice_lr", "E9FLAT/pprice_lr", "E9/baseline_ud"],
}


WF_LAST_DAY = "2026-10-29"   # วันสุดท้ายของ dataset walk-forward รอบที่ 3 (holdout = 10 วันท้าย 14–29 ต.ค.)


def base_rules(rules):
    """นิยาม label ของรอบที่ 3 (ก่อนรอบที่ 4)"""
    r = copy.deepcopy(rules)
    r["label"]["p_ref_morning"], r["label"]["price_1630"] = "last_bar_close", "last_trade_at_or_before"
    return r


def variant_rules(rules, name):
    """pcoff = รอบที่ 3 + P(c) เช้าราคาปิดทางการ · atc = pcoff (ผ่าน T4 แล้ว) + P(16:30) = ATC"""
    r = base_rules(rules)
    if name not in RUN:
        raise SystemExit(f"ไม่รู้จัก {name}")
    r["label"]["p_ref_morning"] = "official_close"
    if name == "atc":
        r["label"]["price_1630"] = "closing_auction"
    return r


def wf_end(rules):
    """dataset ฐานของรอบที่ 3 ที่ตัดวันเดียวกัน → (ฐาน, เวลาตัด)"""
    end = pd.Timestamp(WF_LAST_DAY).tz_localize(rules["timezone"]) + pd.Timedelta(hours=23, minutes=59)
    return load_or_build_dataset(base_rules(rules), as_of=end, tag="_wf"), end


def check_same(base, new, name):
    """แถวเดียวกัน · สิ่งที่นิยามนี้ไม่ควรกระทบต้องเท่าเดิมทุกค่า"""
    b, n = base.sort_values(KEYS).reset_index(drop=True), new.sort_values(KEYS).reset_index(drop=True)
    assert (b[KEYS].astype(str).to_numpy() == n[KEYS].astype(str).to_numpy()).all(), "แถวไม่ตรงกับ dataset เดิม"
    fixed = [c for c in b if c.startswith(("f_", "fd_", "fh_"))]
    if name == "pcoff":
        fixed += ["p_start", "p_end", "r", "direction", "up"]
    else:   # ATC: features ที่คิดจาก r ของรอบก่อน ๆ (f_r_same_*, f_r_prev_*) เปลี่ยนตามนิยาม label โดยตั้งใจ
        changed = [c for c in fixed if "_r_same_mean" in c or "_r_prev_" in c]
        print(f"  ATC: features ที่คิดจาก r ในอดีต {len(changed)} คอลัมน์เปลี่ยนตาม label")
        fixed = [c for c in fixed if c not in changed] + ["p_start"]
    am = (b["session"] == "afternoon").to_numpy()
    for c in fixed:
        x, y = b[c].astype(float).to_numpy(), n[c].astype(float).to_numpy()
        assert np.allclose(x, y, equal_nan=True), f"{c} เปลี่ยน"
    rows = b["session"] == "morning" if name == "pcoff" else b["session"] == "afternoon"
    col = "p_ref" if name == "pcoff" else "p_end"
    same = np.isclose(b[col], n[col], equal_nan=True)
    if name == "pcoff":
        assert same[am].all(), "P(c) รอบบ่ายต้องเท่าเดิม"
    print(f"  ตรวจแล้ว: {len(n)} แถว, {len(fixed)} คอลัมน์ไม่เปลี่ยน; {col} เปลี่ยน {(~same[rows.to_numpy()]).sum()}"
          f" จาก {rows.sum()} แถว{' (เช้า)' if name == 'pcoff' else ' (บ่าย)'}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pcoff"
    rules = load_rules()
    base, end = wf_end(rules)
    r = variant_rules(rules, name)
    ds = load_or_build_dataset(r, as_of=end, tag="_wf")
    print(f"{name}: {ds['date'].min().date()} – {ds['date'].max().date()}, {len(ds)} แถว")
    check_same(base, ds, name)
    if name == "pcoff":
        src = ds.loc[ds["session"] == "morning", "p_ref_source"].value_counts()
        print("  ที่มา P(c) รอบเช้า:", src.to_dict())
    out = OUTPUT_DIR / "experiments" / f"R4{name.upper()}"
    out.mkdir(parents=True, exist_ok=True)
    exps = load_experiments()
    for spec in RUN[name]:
        code, vname = spec.split("/")
        v = next(v for v in exps[code]["variants"] if v["name"] == vname)
        p = run_variant(ds, v, r)
        p.to_csv(out / f"{code}_{vname}_predictions.csv", index=False)
        print(f"  {spec}: {len(p)} แถวทดสอบ ({p['date'].min().date()} – {p['date'].max().date()})", flush=True)


if __name__ == "__main__":
    main()
