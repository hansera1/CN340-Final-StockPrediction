"""E17a (รอบ 7 ข้อ 6): PR-D, PR-D+C, PR-mean — เกณฑ์ลงทะเบียนใน results/e17a_preregistration.md

    python experiments/32_e17a_price.py
→ outputs/experiments/E17a/*_predictions.csv, results/e17a/*.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import round_to_tick, tick_size
from forecast.evaluation.runner import load_or_build_dataset
from forecast.evaluation.splits import walk_forward_splits
from forecast.models.e9 import target_rows
from forecast.models.finetune_lr import MIN_DAYS, PretrainFineTune, logloss

warnings.filterwarnings("ignore", message="All-NaN slice")


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s17, s18, s31 = _module("17_label_variants.py"), _module("18_score_round4.py"), _module("31_e14_finetune.py")
KEYS = ["date", "symbol", "session"]
OUT, RES = OUTPUT_DIR / "experiments" / "E17a", ROOT / "results" / "e17a"
QC = ["q10", "q25", "q50", "q75", "q90"]
CONF = 0.9875


def prepared(ds, exps):
    d, _ = s31.aligned(ds, exps)
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    eq = list(v["aprice_lr"]["features"])
    up_am = v["pprice_lr"]["features"]["morning"]
    up = [c for c in up_am if c != "f_r_prev_afternoon"] + ["t_r_prev1", "t_r_prev2"]
    return d, {"flat_price": (eq, []), "price_updown": (up, s31.PM_ONLY)}


def part(g, target):
    m, y = target_rows(g, target)
    x = g[m]
    return {"X": x, "y": y.astype(int), "dates": x["date"].to_numpy()}


def block(d, cols, session, kind, target, test_dates):
    shared, pm_only = cols
    tr = d[d["date"].to_numpy() < np.min(test_dates)]
    ses = tr[tr["session"] == session]
    pre_cols = shared + ["d_sess_pm", "d_sym_advanc"]
    ft_cols = shared + ["d_sym_advanc"] + (pm_only if session == "afternoon" else [])
    const = {"d_sess_pm": 1.0 if session == "afternoon" else 0.0}
    ft = ses
    if kind == "D+C":
        ft = ses[ses["tick1"]]
        if ft[ft["symbol"] == "KBANK"]["date"].nunique() < MIN_DAYS:
            ft = ft.iloc[:0]
    m = PretrainFineTune()
    if len(ft) == 0:
        m.ft_days_min = 10 ** 9
        m.fit(part(tr, target), part(ses, target), pre_cols, ft_cols, const, inner_val=part(ses, target))
    else:
        m.fit(part(tr, target), part(ft, target), pre_cols, ft_cols, const, inner_val=part(ses, target))
    te = d[(d["session"] == session) & d["date"].isin(test_dates)]
    out = te[KEYS].copy()
    out["p"], out["kappa"], out["inner_ll"] = m.predict(te), m.kappa, m.inner_ll
    return out.assign(target=target, kind=kind)


def tick1_quantiles(d, rules, test_dates, session, symbol):
    """quantile ของ z จากแถวช่องราคา 1 บาทก่อนบล็อก (ไม่พอ 40 วัน → NaN)"""
    tr = d[(d["session"] == session) & (d["symbol"] == symbol) & d["tick1"] & (d["date"].to_numpy() < np.min(test_dates))]
    z = tr["z"].dropna()
    if tr["date"].nunique() < MIN_DAYS:
        return dict.fromkeys(QC, np.nan)
    q = np.quantile(z, rules["quantiles"])
    return dict(zip(QC, q - np.median(z)))


def boot_rmse(ea, eb, dates, n_boot, seed=0):
    """RMSE(a) − RMSE(b): ทุกรอบ bootstrap คำนวณ RMSE ใหม่จากวันที่สุ่มได้ของแต่ละแบบ (จับคู่วันเดียวกัน)"""
    df = pd.DataFrame({"a": ea ** 2, "b": eb ** 2, "d": dates}).groupby("d").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    a, b, n = df["a"].to_numpy(), df["b"].to_numpy(), df["n"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(df), (n_boot, len(df)))
    diff = np.sqrt(a[idx].sum(1) / n[idx].sum(1)) - np.sqrt(b[idx].sum(1) / n[idx].sum(1))
    al = (1 - CONF) / 2
    return np.sqrt(a.sum() / n.sum()) - np.sqrt(b.sum() / n.sum()), np.quantile(diff, al), np.quantile(diff, 1 - al)


GROUPS = [("เช้า", None, "morning"), ("บ่าย", None, "afternoon"), ("KBANK เช้า", "KBANK", "morning"), ("KBANK บ่าย", "KBANK", "afternoon"),
          ("ADVANC เช้า", "ADVANC", "morning"), ("ADVANC บ่าย", "ADVANC", "afternoon")]


def mask(x, sym, sess, tick=False):
    m = (x["session"] == sess).to_numpy()
    if sym:
        m &= (x["symbol"] == sym).to_numpy()
    if tick:
        m &= x["tick1"].to_numpy()
    return m


def score(x, names, n_boot):
    pe, pr = x["p_end"].to_numpy(float), x["p_ref"].to_numpy(float)
    rows = []
    for name in names:
        pt = x[f"pt_{name}"].to_numpy(float)
        for span, tick in [("ทั้งช่วง", False), ("ช่องราคา 1 บาท", True)]:
            for label, sym, sess in GROUPS:
                m = mask(x, sym, sess, tick)
                if not m.any():
                    continue
                e, base, dates = pe[m] - pt[m], pe[m] - x["pt_PR-A"].to_numpy(float)[m], x["date"].to_numpy()[m]
                moved = pt[m] != pr[m]
                right = (np.sign(pt[m] - pr[m]) == np.sign(pe[m] - pr[m]))[moved]
                r = {"model": name, "ช่วง": span, "group": label, "n": int(m.sum()), "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()),
                     "share_moved": moved.mean(), "moved_right": right.mean() if len(right) else np.nan,
                     "exact_tick": np.isclose(np.asarray(round_to_tick(pt[m]), float), pe[m]).mean()}
                if name != "PR-A":
                    r.update(zip(["d_mae", "d_mae_lo", "d_mae_hi"], s31.boot_ci(np.abs(e) - np.abs(base), dates, CONF, n_boot)))
                    r.update(zip(["d_rmse", "d_rmse_lo", "d_rmse_hi"], boot_rmse(e, base, dates, n_boot)))
                rows.append(r)
    return pd.DataFrame(rows)


def coverage(x, points, zq, label):
    rows = []
    pe = x["p_end"].to_numpy(float)
    for name in points:
        pt = x[f"pt_{name}"].to_numpy(float)
        band = {c: np.asarray(round_to_tick(pt * np.exp(zq[c].to_numpy(float) - zq["q50"].to_numpy(float))), float) for c in QC}
        in50, in80 = (pe >= band["q25"]) & (pe <= band["q75"]), (pe >= band["q10"]) & (pe <= band["q90"])
        for lab, sym, sess in GROUPS:
            m = mask(x, sym, sess) & np.isfinite(band["q50"])
            if m.any():
                rows.append({"quantile": label, "model": name, "group": lab, "n": int(m.sum()), "cover50": in50[m].mean(), "cover80": in80[m].mean(),
                             "width50": (band["q75"] - band["q25"])[m].mean(), "width80": (band["q90"] - band["q10"])[m].mean()})
    return pd.DataFrame(rows)


def main():
    rules = load_rules()
    tz, ev = rules["timezone"], rules["evaluation"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    d, cols = prepared(ds, load_experiments())
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    jobs = []
    for session in ["morning", "afternoon"]:
        g = d[d["session"] == session]
        for _, te in walk_forward_splits(g["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]):
            for kind in ["D", "D+C"]:
                for target in cols:
                    jobs.append((session, kind, target, te))
    res = Parallel(n_jobs=-1)(delayed(block)(d, cols[t], s, k, t, te) for s, k, t, te in jobs)
    pr = pd.concat(res, ignore_index=True)
    pr.to_csv(OUT / "ft_predictions.csv", index=False)
    print(len(pr), "แถว · ถึง", pr["date"].max(), flush=True)
    assert str(pr["date"].max())[:10] == "2026-10-12"

    eq, pp, zq = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr", "E9_baseline_ud"]]
    x = d[KEYS + ["p_ref", "p_end", "z", "tick1"]].merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS)
    x = x.merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS).merge(zq[KEYS + QC], on=KEYS)
    for kind in ["D", "D+C"]:
        for target, c in [("flat_price", "eq_"), ("price_updown", "up_")]:
            k = pr[(pr["kind"] == kind) & (pr["target"] == target)][KEYS + ["p"]].rename(columns={"p": c + kind})
            x = x.merge(k, on=KEYS)
    x = x[x["p_end"].notna() & x["p_ref"].notna()].sort_values(KEYS).reset_index(drop=True)
    pref, pe = x["p_ref"].to_numpy(float), x["p_end"].to_numpy(float)

    x["pt_P(c)"] = pref
    x["pt_PR-A"] = s18.price_rule(pref, x["eq_A"].to_numpy(), x["up_A"].to_numpy())[0]
    for kind in ["D", "D+C"]:
        x[f"pt_PR-{kind}"] = s18.price_rule(pref, x[f"eq_{kind}"].to_numpy(), x[f"up_{kind}"].to_numpy())[0]
    # PR-mean: E[z|·] แยกหุ้น/รอบ จากวันก่อนวันที่ทาย (ประวัติเต็มของ dataset) · สูตรเดียวกับ predict.decimal_point
    h = d[KEYS + ["z"]].sort_values(["symbol", "session", "date"]).copy()
    g = h.groupby(["symbol", "session"])["z"]
    pos, neg = h["z"].where(h["z"] > 0), h["z"].where(h["z"] < 0)
    h["z_up"] = pos.groupby([h["symbol"], h["session"]]).transform(lambda s: s.expanding().mean().shift(1))
    h["z_dn"] = neg.groupby([h["symbol"], h["session"]]).transform(lambda s: s.expanding().mean().shift(1))
    x = x.merge(h[KEYS + ["z_up", "z_dn"]], on=KEYS)
    p_hi = (1 - x["eq_A"]) * x["up_A"]
    p_lo = (1 - x["eq_A"]) * (1 - x["up_A"])
    x["pt_PR-mean"] = x["p_ref"] * np.exp(p_hi * x["z_up"] + p_lo * x["z_dn"])
    x["pt_PR-mean-round"] = np.asarray(round_to_tick(x["pt_PR-mean"].to_numpy(float)), float)
    x = x[x["pt_PR-mean"].notna()].reset_index(drop=True)
    print("เทียบบน", len(x), "รอบ", flush=True)

    names = ["PR-A", "P(c)", "PR-D", "PR-D+C", "PR-mean", "PR-mean-round"]
    t = score(x, names, ev["n_boot"])
    t.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    cov = coverage(x, ["PR-A", "PR-D", "PR-D+C", "PR-mean-round"], x[QC], "ทุกแถว")
    tq = []
    for session in ["morning", "afternoon"]:
        g0 = d[d["session"] == session]
        for _, te in walk_forward_splits(g0["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]):
            for sym in rules["symbols"]:
                q = tick1_quantiles(d, rules, te, session, sym)
                tq.append(pd.DataFrame({"date": te, "symbol": sym, "session": session, **{c: q[c] for c in QC}}))
    tq = x[KEYS + ["p_end", "tick1", "pt_PR-A"]].merge(pd.concat(tq), on=KEYS)
    tq = tq[tq["tick1"]].reset_index(drop=True)
    cov = pd.concat([cov, coverage(x[x["tick1"]].reset_index(drop=True), ["PR-A"], x[x["tick1"]].reset_index(drop=True)[QC], "ทุกแถว (เฉพาะแถวช่อง 1 บาท)"),
                     coverage(tq, ["PR-A"], tq[QC], "เฉพาะแถวช่อง 1 บาท")], ignore_index=True)
    cov.to_csv(RES / "coverage.csv", index=False, encoding="utf-8-sig")

    # log loss ของ A-price / p_price
    ll = []
    same = np.isclose(pe, pref, atol=1e-6)
    for name, a, b in [("LR เดิม", "eq_A", "up_A"), ("D", "eq_D", "up_D"), ("D+C", "eq_D+C", "up_D+C")]:
        for label, sym, sess in GROUPS:
            m = mask(x, sym, sess)
            nz = m & ~same
            ll.append({"model": name, "group": label, "ll_eq": logloss(same[m].astype(float), x[a].to_numpy()[m]).mean(),
                       "ll_up": logloss((pe[nz] > pref[nz]).astype(float), x[b].to_numpy()[nz]).mean()})
    pd.DataFrame(ll).to_csv(RES / "logloss.csv", index=False, encoding="utf-8-sig")

    # เกณฑ์ลงทะเบียน
    get = lambda n, span, g: t[(t["model"] == n) & (t["ช่วง"] == span) & (t["group"] == g)].iloc[0]
    cv = cov[cov["quantile"] == "ทุกแถว"]
    gap = lambda n, g, c, tgt: abs(cv[(cv["model"] == n) & (cv["group"] == g)][c].iat[0] - tgt)
    v = []
    for g in ["เช้า", "บ่าย"]:
        for n, main_span, other in [("PR-D", "ทั้งช่วง", "ช่องราคา 1 บาท"), ("PR-D+C", "ช่องราคา 1 บาท", "ทั้งช่วง"), ("PR-mean", "ทั้งช่วง", "ช่องราคา 1 บาท")]:
            w, o = get(n, main_span, g), get(n, other, g)
            win = (w["d_mae_hi"] < 0 and w["d_rmse_lo"] <= 0) or (w["d_rmse_hi"] < 0 and w["d_mae_lo"] <= 0)
            other_ok = o["d_mae_lo"] <= 0 and o["d_rmse_lo"] <= 0
            cn = "PR-mean-round" if n == "PR-mean" else n
            cover_ok = gap(cn, g, "cover50", .5) <= gap("PR-A", g, "cover50", .5) and gap(cn, g, "cover80", .8) <= gap("PR-A", g, "cover80", .8)
            v.append({"รอบ": g, "แบบ": n, "ช่วงหลัก": main_span, "d_mae": w["d_mae"], "mae_lo": w["d_mae_lo"], "mae_hi": w["d_mae_hi"],
                      "d_rmse": w["d_rmse"], "rmse_lo": w["d_rmse_lo"], "rmse_hi": w["d_rmse_hi"], "ชนะ": bool(win), "อีกช่วงไม่แย่": bool(other_ok),
                      "coverage ไม่แย่": bool(cover_ok), "ผ่าน": bool(win and other_ok and cover_ok)})
    v = pd.DataFrame(v)
    v.to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")

    # ตัวแทน: inner-CV log loss รวม A-price + p_price บนบล็อกร่วม
    rep = []
    for sess in ["morning", "afternoon"]:
        p = pr[pr["session"] == sess]
        s = {k: p[p["kind"] == k].groupby(["date", "target"])["inner_ll"].first().unstack().sum(axis=1, min_count=2) for k in ["D", "D+C"]}
        j = pd.concat(s, axis=1).dropna()
        kap = {k: p[(p["kind"] == k)].drop_duplicates(["date", "target"]).groupby("target")["kappa"].agg(lambda c: c.value_counts().to_dict()).to_dict() for k in ["D", "D+C"]}
        rep.append({"ช่อง": f"{sess}_price", "วันร่วม": len(j), "inner_D": j["D"].mean(), "inner_D+C": j["D+C"].mean(),
                    "ตัวแทน": "PR-D" if j["D"].mean() <= j["D+C"].mean() else "PR-D+C", "κ": str(kap)})
    rep = pd.DataFrame(rep)
    rep.to_csv(RES / "representatives.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 260)
    sh = ["model", "ช่วง", "group", "n", "mae", "rmse", "share_moved", "moved_right", "exact_tick", "d_mae", "d_mae_lo", "d_mae_hi", "d_rmse", "d_rmse_lo", "d_rmse_hi"]
    print(t[t["ช่วง"] == "ทั้งช่วง"][sh].round(4).to_string(index=False))
    print("\nคำตัดสิน (เทียบ PR-A, CI 98.75%)\n", v.round(4).to_string(index=False))
    print("\nตัวแทน\n", rep.drop(columns="κ").round(5).to_string(index=False))
    print("\ncoverage\n", cov[cov["group"].isin(["เช้า", "บ่าย"])].round(3).to_string(index=False))
    print("\nlog loss\n", pd.DataFrame(ll).query("group in ['เช้า','บ่าย']").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
