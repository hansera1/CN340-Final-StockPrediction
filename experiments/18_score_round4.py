"""คิดคะแนน NEXT_STEPS รอบที่ 4 จาก predictions walk-forward (ไม่แตะ holdout) — block bootstrap รายวัน 2,000 ครั้ง

    python experiments/18_score_round4.py t4     # ข้อ 4: กฎค่ามัธยฐานด้วย P(c) รอบเช้าแบบใหม่ เทียบ "ราคาจุด = P(c) แบบใหม่"
    python experiments/18_score_round4.py t1     # ข้อ 1ข: 3 คำตอบ (fd_lr + A-dir LR) เทียบ fd_lr ขึ้น/ลง (นับ "เท่าเดิม" ถูกได้)
    python experiments/18_score_round4.py atc    # ข้อ 1ค: fd_lr / กฎราคา บน label 16:30 = ATC (เก็บไว้ ใช้เมื่ออาจารย์ตอบ ATC)
→ results/round4/*.csv
"""
import sys

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_rules
from forecast.evaluation.metrics import round_to_tick, tick_size

EXP = OUTPUT_DIR / "experiments"
OUT = ROOT / "results" / "round4"
KEYS = ["date", "symbol", "session"]
GROUPS = [None, "morning", "afternoon", ("KBANK", "morning"), ("KBANK", "afternoon"), ("ADVANC", "morning"),
          ("ADVANC", "afternoon"), ("KBANK", None), ("ADVANC", None)]
QC = ["q10", "q25", "q50", "q75", "q90"]


def load(exp, variant, tz):
    p = pd.read_csv(EXP / exp / f"{variant}_predictions.csv")
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def gname(g):
    if g is None:
        return "ALL"
    if isinstance(g, str):
        return f"ALL/{g}"
    return f"{g[0]}/{g[1] or 'ALL'}"


def subset(d, g):
    if g is None:
        return d.index
    if isinstance(g, str):
        return d.index[d["session"] == g]
    m = d["symbol"] == g[0]
    if g[1]:
        m &= d["session"] == g[1]
    return d.index[m]


def boot_mean(values, dates, n_boot, seed=0):
    """ค่าเฉลี่ยต่อแถว + 95% CI จาก block bootstrap รายวัน (ทุกแถวของวันเดียวกันไปด้วยกัน)"""
    df = pd.DataFrame({"v": np.asarray(values, float), "d": np.asarray(dates)}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    daily = df.groupby("d")["v"].agg(["sum", "count"])
    s, c = daily["sum"].to_numpy(), daily["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def joined(parts):
    """{ชื่อ: DataFrame} → แถวร่วม (KEYS) เรียงเดียวกัน"""
    keys = None
    for p in parts.values():
        k = p[KEYS]
        keys = k if keys is None else keys.merge(k, on=KEYS)
    return {n: keys.merge(p, on=KEYS, how="left") for n, p in parts.items()}


def price_rule(p_ref, p_eq, p_price):
    up, dn = (1 - p_eq) * p_price, (1 - p_eq) * (1 - p_price)
    shift = np.where(up > 0.5, 1, np.where(dn > 0.5, -1, 0))
    return p_ref + shift * tick_size(p_ref), shift


def score_price(d, eq, pp, zq, n_boot, label):
    """MAE กฎค่ามัธยฐาน vs P(c) + ความครอบคลุมช่วงราคา (แบบวันจริง: point·exp(zq − zq50) ปัดช่องราคา)"""
    p_ref, p_end = d["p_ref"].to_numpy(float), d["p_end"].to_numpy(float)
    point, shift = price_rule(p_ref, eq["p_up"].to_numpy(), pp["p_up"].to_numpy())
    e_rule, e_naive = np.abs(p_end - point), np.abs(p_end - p_ref)
    band = {c: round_to_tick(point * np.exp(zq[c].to_numpy() - zq["q50"].to_numpy())) for c in QC}
    in50 = ((p_end >= band["q25"]) & (p_end <= band["q75"])).astype(float)
    in80 = ((p_end >= band["q10"]) & (p_end <= band["q90"])).astype(float)
    rows = []
    for g in GROUPS:
        m = subset(d, g)
        dm, lo, hi = boot_mean(e_rule[m] - e_naive[m], d.loc[m, "date"], n_boot)
        rows.append({"label": label, "group": gname(g), "n": len(m), "n_days": d.loc[m, "date"].nunique(),
                     "n_shift_up": int((shift[m] == 1).sum()), "n_shift_down": int((shift[m] == -1).sum()),
                     "mae_rule_baht": e_rule[m].mean(), "mae_pc_baht": e_naive[m].mean(),
                     "d_mae_baht": dm, "d_lo": lo, "d_hi": hi,
                     "worse_clearly": bool(lo > 0), "cover50": in50[m].mean(), "cover80": in80[m].mean()})
    return pd.DataFrame(rows)


def t4(rules, n_boot):
    tz = rules["timezone"]
    new = joined({"eq": load("R4PCOFF", "E9FLAT_aprice_lr", tz), "pp": load("R4PCOFF", "E9FLAT_pprice_lr", tz),
                  "zq": load("R4PCOFF", "E9_baseline_ud", tz)})
    old = joined({"eq": load("E9FLAT", "aprice_lr", tz), "pp": load("E9FLAT", "pprice_lr", tz),
                  "zq": load("E9", "baseline_ud", tz)})
    keys = new["eq"][KEYS].merge(old["eq"][KEYS], on=KEYS)
    lab = new["eq"][KEYS + ["p_ref", "p_end"]]
    ok = keys.merge(lab, on=KEYS)
    ok = ok[ok["p_end"].notna()][KEYS]
    new = {k: ok.merge(v, on=KEYS, how="left") for k, v in new.items()}
    old = {k: ok.merge(v, on=KEYS, how="left") for k, v in old.items()}
    d = new["eq"]
    print(f"T4: {len(d)} รอบ ({d['date'].min().date()} – {d['date'].max().date()})")
    a = score_price(d, new["eq"], new["pp"], new["zq"], n_boot, "ใหม่: โมเดลราคาฝึกด้วย P(c) ใหม่")
    # อ้างอิง: โมเดลราคาเดิม (ฝึกด้วย P(c) เดิม) แต่วัดรอบ P(c) ใหม่ = สิ่งที่วันจริงทำอยู่ตอนนี้
    od = d[KEYS + ["p_ref", "p_end"]]
    b = score_price(od, old["eq"], old["pp"], old["zq"], n_boot, "เดิม: โมเดลราคาฝึกด้วย P(c) เดิม ใช้กับ P(c) ใหม่")
    tab = pd.concat([a, b], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "t4_price_official_pref.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))


def t1(rules, n_boot):
    tz = rules["timezone"]
    p = joined({"p": load("E12", "fd_lr", tz), "f": load("E9FLAT", "adir_lr", tz)})
    d, f = p["p"], p["f"]["p_up"].to_numpy()
    m = d["direction"].notna().to_numpy()
    d, f = d[m].reset_index(drop=True), f[m]
    pu, direction = d["p_up"].to_numpy(), d["direction"].to_numpy()
    probs = np.column_stack([(1 - f) * pu, f, (1 - f) * (1 - pu)])
    three = np.array([1, 0, -1])[probs.argmax(1)]
    two = np.where(pu >= 0.5, 1, -1)
    c3, c2 = (three == direction).astype(float), (two == direction).astype(float)
    rows = []
    for g in GROUPS:
        ix = subset(d, g)
        dm, lo, hi = boot_mean(c3[ix] - c2[ix], d.loc[ix, "date"], n_boot)
        rows.append({"group": gname(g), "n": len(ix), "acc_3way": c3[ix].mean(), "acc_updown_fw": c2[ix].mean(),
                     "share_flat_answer": (three[ix] == 0).mean(), "flat_rate": (direction[ix] == 0).mean(),
                     "d_3way_minus_updown": dm, "d_lo": lo, "d_hi": hi, "use_if_allowed": bool(g is None and lo > 0)})
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "t1_three_way.csv", index=False, encoding="utf-8-sig")
    print(f"T1: {len(d)} รอบ")
    print(tab.round(4).to_string(index=False))


def atc(rules, n_boot):
    tz = rules["timezone"]
    p = joined({"m": load("R4ATC", "E12_fd_lr", tz), "b": load("R4ATC", "E12_baseline_ud", tz),
                "eq": load("R4ATC", "E9FLAT_aprice_lr", tz), "pp": load("R4ATC", "E9FLAT_pprice_lr", tz),
                "zq": load("R4ATC", "E9_baseline_ud", tz)})
    d = p["m"]
    ok = (d["direction"].notna() & d["p_end"].notna()).to_numpy()
    p = {k: v[ok].reset_index(drop=True) for k, v in p.items()}
    d = p["m"]
    nf = (d["direction"] != 0).to_numpy()
    rows = []
    for name, x in [("fd_lr (ATC)", p["m"]), ("E0-ud (ATC)", p["b"])]:
        cor = ((x["p_up"] >= 0.5).to_numpy() == (d["direction"] == 1).to_numpy()).astype(float)
        ref = ((p["b"]["p_up"] >= 0.5).to_numpy() == (d["direction"] == 1).to_numpy()).astype(float)
        for g in GROUPS:
            ix = subset(d, g)
            ixn = ix[nf[ix]]
            r = {"model": name, "group": gname(g), "n_nonflat": len(ixn), "acc_nf": cor[ixn].mean(),
                 "acc_fw": (cor[ix] * nf[ix]).mean(), "flat_rate": 1 - nf[ix].mean()}
            if name.startswith("fd_lr"):
                r.update(dict(zip(["d_vs_e0ud", "d_lo", "d_hi"], boot_mean(cor[ixn] - ref[ixn],
                                                                           d.loc[ixn, "date"], n_boot))))
            rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "atc_direction.csv", index=False, encoding="utf-8-sig")
    price = score_price(d, p["eq"], p["pp"], p["zq"], n_boot, "ATC")
    price.to_csv(OUT / "atc_price.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"ATC: {len(d)} รอบ")
    print(tab.round(4).to_string(index=False))
    print(price.round(4).to_string(index=False))


if __name__ == "__main__":
    rules = load_rules()
    {"t4": t4, "t1": t1, "atc": atc}[sys.argv[1]](rules, rules["evaluation"]["n_boot"])
