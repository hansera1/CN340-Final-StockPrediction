"""NEXT_STEPS รอบที่ 5 ข้อ B — ราคาไม่ปัดช่อง (เตรียมไว้ ใช้เฉพาะถ้าอาจารย์ให้ส่งทศนิยมได้) บน walk-forward เดิม

    python experiments/22_b_decimal_price.py
→ results/round5/b_decimal_price.csv, b_direction_share.csv

ราคา = P(c)·exp(P(สูงกว่า)·E[z|สูงกว่า] + P(ต่ำกว่า)·E[z|ต่ำกว่า]) · E[z|·] จากแถวที่รู้ผลแล้วของ (หุ้น, รอบ) เดียวกัน
ก่อนวันที่ทาย (= สิ่งที่วันจริงทำ) · P(สูงกว่า/ต่ำกว่า) จาก A-price LR + p_price ที่ฝึกด้วย P(c) แบบใหม่ (R4PCOFF)
เทียบ MAE กับกฎค่ามัธยฐาน (ปัดช่อง) และราคาจุด = P(c) · block bootstrap รายวัน 2,000 ครั้ง · ไม่แตะ holdout
"""
import importlib

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import ROOT, load_rules
from forecast.evaluation.runner import load_or_build_dataset

s18 = importlib.import_module("18_score_round4")
s17 = importlib.import_module("17_label_variants")
OUT = ROOT / "results" / "round5"


def main():
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    _, end = s17.wf_end(rules)
    ds = load_or_build_dataset(s17.variant_rules(rules, "pcoff"), as_of=end, tag="_wf")
    p = s18.joined({"eq": s18.load("R4PCOFF", "E9FLAT_aprice_lr", tz), "pp": s18.load("R4PCOFF", "E9FLAT_pprice_lr", tz)})
    d = p["eq"][s18.KEYS + ["p_ref", "p_end"]].copy()
    ok = d["p_end"].notna().to_numpy()
    d, eq, pp = d[ok].reset_index(drop=True), p["eq"]["p_up"].to_numpy()[ok], p["pp"]["p_up"].to_numpy()[ok]
    up, dn = (1 - eq) * pp, (1 - eq) * (1 - pp)

    # E[z | z > 0], E[z | z < 0] แบบ expanding: เฉพาะวันที่ < วันที่ทาย ของ (หุ้น, รอบ) เดียวกัน
    h = ds[ds["z"].notna()].sort_values("date")
    zu, zd = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for (sym, sess), g in d.groupby(["symbol", "session"]):
        hh = h[(h["symbol"] == sym) & (h["session"] == sess)]
        for i, dt in zip(g.index, g["date"]):
            z = hh.loc[hh["date"] < dt, "z"]
            zu[i], zd[i] = z[z > 0].mean(), z[z < 0].mean()
    p_ref, p_end = d["p_ref"].to_numpy(float), d["p_end"].to_numpy(float)
    dec = p_ref * np.exp(up * zu + dn * zd)
    rule, _ = s18.price_rule(p_ref, eq, pp)
    e_dec, e_rule, e_pc = np.abs(p_end - dec), np.abs(p_end - rule), np.abs(p_end - p_ref)
    rows = []
    for g in s18.GROUPS:
        m = s18.subset(d, g)
        dt = d.loc[m, "date"]
        r = {"group": s18.gname(g), "n": len(m), "mae_decimal": e_dec[m].mean(), "mae_rule": e_rule[m].mean(),
             "mae_pc": e_pc[m].mean(), "mean_abs_shift_baht": np.abs(dec[m] - p_ref[m]).mean()}
        for k, base in [("rule", e_rule), ("pc", e_pc)]:
            r.update(dict(zip([f"d_vs_{k}", f"d_vs_{k}_lo", f"d_vs_{k}_hi"],
                              s18.boot_mean(e_dec[m] - base[m], dt, n_boot))))
        rows.append(r)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "b_decimal_price.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"B: {len(d)} รอบ ({d['date'].min().date()} – {d['date'].max().date()})")
    print(t.round(4).to_string(index=False))

    # สำหรับรายงาน: สัดส่วนที่ fd_lr ทาย "ลง" และสัดส่วนวันลงในวันไม่ flat
    f = s18.load("E12", "fd_lr", tz)
    lab = ds[ds["direction"].notna() & (ds["direction"] != 0) & (ds["date"] < f["date"].min())]
    rows = []
    for (sym, sess), g in f.groupby(["symbol", "session"]):
        nf = g[g["direction"].notna() & (g["direction"] != 0)]
        first = lab[(lab["symbol"] == sym) & (lab["session"] == sess)]
        allnf = ds[(ds["symbol"] == sym) & (ds["session"] == sess) & ds["direction"].notna() & (ds["direction"] != 0)
                   & (ds["date"] <= f["date"].max())]
        rows.append({"symbol": sym, "session": sess, "n_pred": len(g), "share_pred_down": (g["p_up"] < 0.5).mean(),
                     "share_down_nonflat_test": (nf["direction"] == -1).mean(),
                     "share_down_nonflat_first_train": (first["direction"] == -1).mean(),
                     "share_down_nonflat_all_wf": (allnf["direction"] == -1).mean()})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "b_direction_share.csv", index=False, encoding="utf-8-sig")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
