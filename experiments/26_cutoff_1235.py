"""อาจารย์ยืนยัน (6 พ.ย.): รอบบ่ายส่งก่อน 13:00, ราคาปิดบ่าย = ราคาปิดทางการหลัง ATC → ย้ายจุดตัดบ่าย 12:15 → 12:35 (หลังตลาดเช้าปิด)

รันโมเดลที่ล็อก (สเปกเดิม ไม่จูน) บน walk-forward เดิม แถวเดิม ข้อมูลตัดที่ 29/10/2026 ไม่แตะ holdout:
  - 12:15 (ATC) = outputs/experiments/R4ATC/ (รอบที่ 4)
  - 12:35 (ATC) = outputs/experiments/C1235/
ตรวจก่อนรันว่ารอบเช้าไม่เปลี่ยน และรอบบ่ายเปลี่ยนเฉพาะ features ภายในวัน / P(c)

    python experiments/26_cutoff_1235.py
→ results/cutoff_1235/direction.csv, price.csv
"""
import copy
import importlib.util

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset, run_variant


def _module(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("/", "_"), ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s17, s18 = _module("17_label_variants.py"), _module("18_score_round4.py")
KEYS, GROUPS, RESULTS = s18.KEYS, s18.GROUPS, ROOT / "results" / "cutoff_1235"
CUT, OLD_CUT = "12:35", "12:15"
RUN = s17.RUN["atc"]


def with_cutoff(rules, cut):
    r = s17.variant_rules(rules, "atc")
    r["sessions"] = copy.deepcopy(r["sessions"])
    r["sessions"]["afternoon"]["cutoff"] = cut
    return r


def check(old, new):
    """แถวเดียวกัน · รอบเช้าเท่าเดิมทุกค่า · รอบบ่าย label ทิศทาง / P(ปลาย) เท่าเดิม → เปลี่ยนเฉพาะสิ่งที่รู้ ณ จุดตัด"""
    a, b = old.sort_values(KEYS).reset_index(drop=True), new.sort_values(KEYS).reset_index(drop=True)
    assert (a[KEYS].astype(str).to_numpy() == b[KEYS].astype(str).to_numpy()).all(), "แถวไม่ตรง"
    pm = (a["session"] == "afternoon").to_numpy()
    cols = [c for c in a if c.startswith(("f_", "fd_", "fh_", "fl_"))]
    changed = [c for c in cols if not np.allclose(a.loc[pm, c].astype(float), b.loc[pm, c].astype(float), equal_nan=True)]
    for c in cols + ["p_ref", "p_start", "p_end"]:
        assert np.allclose(a.loc[~pm, c].astype(float), b.loc[~pm, c].astype(float), equal_nan=True), f"รอบเช้า {c} เปลี่ยน"
    for c in ["p_start", "p_end", "r", "direction", "up"]:
        assert np.allclose(a.loc[pm, c].astype(float), b.loc[pm, c].astype(float), equal_nan=True), f"label บ่าย {c} เปลี่ยน"
    same_ref = np.isclose(a.loc[pm, "p_ref"], b.loc[pm, "p_ref"], equal_nan=True)
    print(f"  ตรวจแล้ว: {len(b)} แถว · รอบเช้าเท่าเดิม · label บ่ายเท่าเดิม · P(c) บ่ายเปลี่ยน {(~same_ref).sum()}/{pm.sum()} แถว")
    print(f"  features รอบบ่ายที่เปลี่ยน {len(changed)} ตัว: {changed}")


def score(rules, n_boot):
    tz = rules["timezone"]
    names = {"m": "E12_fd_lr", "b": "E12_baseline_ud", "eq": "E9FLAT_aprice_lr", "pp": "E9FLAT_pprice_lr", "zq": "E9_baseline_ud"}
    parts = {}
    for tag, exp in [("old", "R4ATC"), ("new", "C1235")]:
        for k, v in names.items():
            parts[f"{tag}_{k}"] = s18.load(exp, v, tz)
    p = s18.joined(parts)
    d = p["new_m"]
    ok = (d["direction"].notna() & d["p_end"].notna()).to_numpy()
    p = {k: v[ok].reset_index(drop=True) for k, v in p.items()}
    d = p["new_m"]
    nf = (d["direction"] != 0).to_numpy()
    y = (d["direction"] == 1).to_numpy()
    cor = {k: ((p[f"{k}_m"]["p_up"] >= 0.5).to_numpy() == y).astype(float) for k in ["old", "new"]}
    base = {k: ((p[f"{k}_b"]["p_up"] >= 0.5).to_numpy() == y).astype(float) for k in ["old", "new"]}
    rows = []
    for g in GROUPS:
        ix = s18.subset(d, g)
        n = ix[nf[ix]]
        dates = d.loc[n, "date"]
        rows.append({"group": s18.gname(g), "n_nonflat": len(n), "flat_rate": 1 - nf[ix].mean(),
                     "acc_1215": cor["old"][n].mean(), "acc_1235": cor["new"][n].mean(), "e0ud_1235": base["new"][n].mean(),
                     **dict(zip(["d_new_vs_old", "d1_lo", "d1_hi"], s18.boot_mean(cor["new"][n] - cor["old"][n], dates, n_boot))),
                     **dict(zip(["d_new_vs_e0ud", "d2_lo", "d2_hi"], s18.boot_mean(cor["new"][n] - base["new"][n], dates, n_boot))),
                     "acc_fw_1215": (cor["old"][ix] * nf[ix]).mean(), "acc_fw_1235": (cor["new"][ix] * nf[ix]).mean()})
    tab = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / "direction.csv", index=False, encoding="utf-8-sig")
    price = pd.concat([s18.score_price(p["new_m"], p["new_eq"], p["new_pp"], p["new_zq"], n_boot, CUT),
                       s18.score_price(p["old_m"], p["old_eq"], p["old_pp"], p["old_zq"], n_boot, OLD_CUT)], ignore_index=True)
    price.to_csv(RESULTS / "price.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"\nเทียบบน {len(d)} รอบ ({d['date'].min().date()} – {d['date'].max().date()})")
    print(tab.round(4).to_string(index=False))
    print(price[["label", "group", "n", "mae_rule_baht", "mae_pc_baht", "d_mae_baht", "d_lo", "d_hi", "cover50", "cover80"]].round(4).to_string(index=False))


def main():
    rules = load_rules()
    rules = copy.deepcopy(rules)
    rules["sessions"]["afternoon"]["cutoff"] = OLD_CUT
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(rules["timezone"]) + pd.Timedelta(hours=23, minutes=59)
    old = load_or_build_dataset(with_cutoff(rules, OLD_CUT), as_of=end, tag="_wf")
    r = with_cutoff(rules, CUT)
    ds = load_or_build_dataset(r, as_of=end, tag="_wf", rebuild=True)
    print(f"ATC + จุดตัดบ่าย {CUT}: {ds['date'].min().date()} – {ds['date'].max().date()}, {len(ds)} แถว")
    check(old, ds)
    out = OUTPUT_DIR / "experiments" / "C1235"
    out.mkdir(parents=True, exist_ok=True)
    exps = load_experiments()
    for spec in RUN:
        code, vname = spec.split("/")
        v = next(v for v in exps[code]["variants"] if v["name"] == vname)
        p = run_variant(ds, v, r)
        p.to_csv(out / f"{code}_{vname}_predictions.csv", index=False)
        print(f"  {spec}: {len(p)} แถวทดสอบ ({p['date'].min().date()} – {p['date'].max().date()})", flush=True)
    score(rules, rules["evaluation"]["n_boot"])


if __name__ == "__main__":
    main()
