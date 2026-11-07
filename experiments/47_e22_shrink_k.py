"""E22: หดราคาทายเข้าหาราคาอ้างอิงด้วยค่า k — เกณฑ์ใน results/e22_preregistration.md · รายงานเท่านั้น ไม่แตะระบบ live

    python experiments/47_e22_shrink_k.py
→ outputs/experiments/E22/ (มีราคาจริง ไม่อยู่ใน git) · results/e22/ (ตัวเลขสรุปเท่านั้น)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset, run_variant
from forecast.evaluation.splits import walk_forward_splits
from forecast.models.e9 import inner_folds

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
KS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_OOS_DAYS = 40
CONF, CONF95 = 0.997, 0.95
QC = ["q10", "q25", "q50", "q75", "q90"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
RES, OUT = ROOT / "results" / "e22", OUTPUT_DIR / "experiments" / "E22"
FIXED = {f"k={k:g}": k for k in KS}


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def expected_z(p_eq, p_price, z_up, z_dn):
    return (1 - p_eq) * p_price * z_up + (1 - p_eq) * (1 - p_price) * z_dn


def shrink_point(p_ref, ez, k):
    return p_ref * np.exp(k * ez)


def select_k(p_ref, p_end, ez, dates, ks=KS, min_days=MIN_OOS_DAYS):
    """k ที่ RMSE เฉลี่ยของ 3 บล็อกท้ายต่ำสุด · ใช้เฉพาะแถวที่ส่งเข้ามา · ค่าเท่ากันเลือก k น้อยกว่า"""
    if len(np.unique(dates)) < min_days:
        return 0.0
    folds = [va for _, va in inner_folds(dates)]
    if not folds:
        return 0.0
    scores = [np.mean([np.sqrt(np.mean((p_end[va] - shrink_point(p_ref[va], ez[va], k)) ** 2)) for va in folds]) for k in ks]
    return float(ks[int(np.argmin(np.round(scores, 10)))])


def walk_forward_k(x, blocks):
    """k ต่อแถว: แต่ละบล็อกเลือกจากแถววันก่อนบล็อกเท่านั้น"""
    out = pd.Series(np.nan, index=x.index)
    dates, pr, pe, ez = x["date"].to_numpy(), x["p_ref"].to_numpy(float), x["p_end"].to_numpy(float), x["ez"].to_numpy(float)
    for te in blocks:
        start = np.min(te)
        before, now = dates < start, np.isin(dates, te)
        if now.any():
            out[now] = select_k(pr[before], pe[before], ez[before], dates[before])
    return out


def cond_means(d):
    """E[z|·] แยกหุ้นและรอบ จากวันก่อนวันทาย"""
    h = d[KEYS + ["z"]].sort_values(["symbol", "session", "date"]).copy()
    grp = [h["symbol"], h["session"]]
    h["z_up"] = h["z"].where(h["z"] > 0).groupby(grp).transform(lambda s: s.expanding().mean().shift(1))
    h["z_dn"] = h["z"].where(h["z"] < 0).groupby(grp).transform(lambda s: s.expanding().mean().shift(1))
    return h[KEYS + ["z_up", "z_dn"]]


def attach_points(x, s18):
    pr = x["p_ref"].to_numpy(float)
    x["ez"] = expected_z(x["eq_A"].to_numpy(), x["up_A"].to_numpy(), x["z_up"].to_numpy(), x["z_dn"].to_numpy())
    x["pt_PR-A"] = s18.price_rule(pr, x["eq_A"].to_numpy(), x["up_A"].to_numpy())[0]
    x["pt_P(c)"] = pr
    for name, k in FIXED.items():
        x[f"pt_{name}"] = shrink_point(pr, x["ez"].to_numpy(), k)
    return x


def groups(x):
    out = []
    for sess, th in SESSIONS:
        m = (x["session"] == sess).to_numpy()
        out.append((th, m))
        for sym in ("KBANK", "ADVANC"):
            out.append((f"{sym} {th}", m & (x["symbol"] == sym).to_numpy()))
    return out


def score(x, names, n_boot, boot_rmse, boot_ci):
    pe, dates, rows = x["p_end"].to_numpy(float), x["date"].to_numpy(), []
    for name in names:
        e = pe - x[f"pt_{name}"].to_numpy(float)
        for lab, m in groups(x):
            if not m.any():
                continue
            r = {"model": name, "group": lab, "n": int(m.sum()), "n_days": len(np.unique(dates[m])), "rmse": np.sqrt((e[m] ** 2).mean()), "mae": np.abs(e[m]).mean()}
            for base, tag in [("PR-A", "pra"), ("P(c)", "pc")]:
                if name == base:
                    continue
                b = pe - x[f"pt_{base}"].to_numpy(float)
                r.update(zip([f"d_rmse_{tag}", f"d_rmse_{tag}_lo", f"d_rmse_{tag}_hi"], boot_rmse(e[m], b[m], dates[m], CONF, n_boot)))
                r.update(zip([f"d_mae_{tag}", f"d_mae_{tag}_lo", f"d_mae_{tag}_hi"], boot_ci(np.abs(e[m]) - np.abs(b[m]), dates[m], CONF, n_boot)))
                r.update(zip([f"d95_rmse_{tag}_lo", f"d95_rmse_{tag}_hi"], boot_rmse(e[m], b[m], dates[m], CONF95, n_boot)[1:]))
            rows.append(r)
    return pd.DataFrame(rows)


def verdict(t):
    q, rows = t.set_index(["model", "group"]), []
    for _, th in SESSIONS:
        a = q.loc[("PR-k", th)]
        k1 = bool(a["d_rmse_pra_hi"] < 0)
        k2 = all(q.loc[("PR-k", f"{s} {th}"), "d_rmse_pra_lo"] <= 0 for s in ("KBANK", "ADVANC"))
        k3 = bool(a["d_mae_pra_lo"] <= 0)
        rows.append({"session": th, "c1_rmse": k1, "c2_stocks": k2, "c3_mae": k3, "pass": k1 and k2 and k3})
    return pd.DataFrame(rows)


def coverage(x):
    pe, rows = x["p_end"].to_numpy(float), []
    for name in ["PR-A", "P(c)", "PR-k"]:
        pt = x[f"pt_{name}"].to_numpy(float)
        band = {c: np.round(pt * np.exp(x[c].to_numpy(float) - x["q50"].to_numpy(float)), 2) for c in QC}
        for lab, m in groups(x):
            rows.append({"model": name, "group": lab, "cover50": ((pe >= band["q25"]) & (pe <= band["q75"]))[m].mean(), "cover80": ((pe >= band["q10"]) & (pe <= band["q90"]))[m].mean()})
    return pd.DataFrame(rows)


def report_window(ds, days, exps, rules, k_by_session, s18):
    """holdout / ข้อมูลใหม่: โมเดลราคาฝึกก่อนช่วงนี้ · k จากผล walk-forward · รายงานเท่านั้น"""
    r = copy.deepcopy(rules)
    r["evaluation"]["holdout_days"] = days
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    eq, up = run_variant(ds, v["aprice_lr"], r, holdout_eval=True), run_variant(ds, v["pprice_lr"], r, holdout_eval=True)
    x = eq[KEYS + ["p_ref", "p_end", "p_up"]].rename(columns={"p_up": "eq_A"}).merge(up[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS).merge(cond_means(ds), on=KEYS)
    x = x[x["p_end"].notna() & x["p_ref"].notna()].sort_values(KEYS).reset_index(drop=True)
    x = attach_points(x, s18)
    x["k"] = x["session"].map(k_by_session)
    x["pt_PR-k"] = shrink_point(x["p_ref"].to_numpy(float), x["ez"].to_numpy(float), x["k"].to_numpy(float))
    return x


def main():
    s17, s18, s31, s38 = [_module(n) for n in ["17_label_variants.py", "18_score_round4.py", "31_e14_finetune.py", "38_e18b_gap_shift.py"]]
    rules = load_rules()
    assert rules["label"]["p_ref_morning"] == "official_close" and rules["label"]["price_1630"] == "closing_auction" and rules["sessions"]["afternoon"]["cutoff"] == "12:35"
    tz, ev = rules["timezone"], rules["evaluation"]
    n_boot = ev["n_boot"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    exps = load_experiments()
    d, _ = s31.aligned(ds, exps)
    RES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    eq, pp, zq = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr", "E9_baseline_ud"]]
    x = d[KEYS + ["p_ref", "p_end"]].merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS).merge(zq[KEYS + QC], on=KEYS)
    x = x.merge(cond_means(d), on=KEYS)
    x = x[x["p_end"].notna() & x["p_ref"].notna() & x["z_up"].notna() & x["z_dn"].notna()].sort_values(KEYS).reset_index(drop=True)
    x = attach_points(x, s18)

    for sess, th in SESSIONS:
        m = x["session"] == sess
        rm = lambda c: round(float(np.sqrt(((x.loc[m, "p_end"] - x.loc[m, c]) ** 2).mean())), 3)
        print(th, int(m.sum()), "แถว · PR-A", rm("pt_PR-A"), "· P(c)", rm("pt_P(c)"), "· k=1", rm("pt_k=1"), flush=True)
    m_am, m_pm = x["session"] == "morning", x["session"] == "afternoon"
    rmse = lambda m, c: round(float(np.sqrt(((x.loc[m, "p_end"] - x.loc[m, c]) ** 2).mean())), 3)
    got = (rmse(m_am, "pt_PR-A"), rmse(m_pm, "pt_PR-A"), rmse(m_am, "pt_P(c)"), rmse(m_pm, "pt_P(c)"), rmse(m_am, "pt_k=1"), rmse(m_pm, "pt_k=1"))
    assert got == (3.083, 1.830, 3.073, 1.817, 3.061, 1.815), f"ไม่ตรงกับ E17a: {got}"

    x["k"] = np.nan
    k_last = {}
    for sess, _ in SESSIONS:
        m = (x["session"] == sess).to_numpy()
        g = x[m].reset_index(drop=True)
        days = d.loc[d["session"] == sess, "date"].to_numpy()
        blocks = [te for _, te in walk_forward_splits(days, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
        x.loc[m, "k"] = walk_forward_k(g, blocks).to_numpy()
        k_last[sess] = select_k(g["p_ref"].to_numpy(float), g["p_end"].to_numpy(float), g["ez"].to_numpy(float), g["date"].to_numpy())
    assert x["k"].notna().all()
    x["pt_PR-k"] = shrink_point(x["p_ref"].to_numpy(float), x["ez"].to_numpy(float), x["k"].to_numpy(float))
    x.to_csv(OUT / "points.csv", index=False)

    names = ["PR-A", "P(c)", "PR-k"] + [n for n in FIXED if n != "k=0"]
    t = score(x, names, n_boot, s38.boot_rmse, s31.boot_ci)
    v = verdict(t)
    cov = coverage(x)
    picks = x.groupby(["session", "k"]).agg(n=("k", "size"), n_days=("date", "nunique")).reset_index()
    t.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    v.to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(RES / "coverage.csv", index=False, encoding="utf-8-sig")
    picks.to_csv(RES / "k_selected.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([k_last]).to_csv(RES / "k_last.csv", index=False, encoding="utf-8-sig")

    ho = report_window(ds, 10, exps, rules, k_last, s18)
    ud = np.sort(ds["date"].unique())
    assert str(ud[-10])[:10] == "2026-10-14" and str(ud[-1])[:10] == "2026-10-29"
    new_ds = load_or_build_dataset(rules, tag="_new")
    nd = np.sort(new_ds["date"].unique())
    assert str(nd[-6])[:10] == "2026-10-30" and str(nd[-1])[:10] == "2026-11-06", nd[-7:]
    nw = report_window(new_ds, 6, exps, rules, k_last, s18)
    nw.to_csv(OUT / "new_rows.csv", index=False)
    wnames = ["PR-A", "P(c)", "PR-k", "k=1"]
    ho_t, nw_t = score(ho, wnames, n_boot, s38.boot_rmse, s31.boot_ci), score(nw, wnames, n_boot, s38.boot_rmse, s31.boot_ci)
    ho_t.to_csv(RES / "holdout.csv", index=False, encoding="utf-8-sig")
    nw_t.to_csv(RES / "new_data.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    show = ["model", "group", "n", "rmse", "mae", "d_rmse_pra", "d_rmse_pra_lo", "d_rmse_pra_hi", "d95_rmse_pra_lo", "d95_rmse_pra_hi", "d_mae_pra", "d_mae_pra_lo", "d_mae_pra_hi", "d_rmse_pc", "d_rmse_pc_lo", "d_rmse_pc_hi"]
    print(t[show].round(3).to_string(index=False))
    print(v.to_string(index=False))
    print(picks.to_string(index=False))
    print("k ล่าสุด", k_last)
    print(cov.round(3).to_string(index=False))
    for tag, tab in [("holdout", ho_t), ("ข้อมูลใหม่", nw_t)]:
        print("---", tag)
        print(tab[["model", "group", "n", "rmse", "mae", "d_rmse_pra", "d_rmse_pra_lo", "d_rmse_pra_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
