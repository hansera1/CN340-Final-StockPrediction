"""E14 / E17a ส่วนเพิ่ม: C′, C-adv, PR-C′ — สเปกเพื่อน (spec_E14_Cprime_Cadv) · เกณฑ์ใน results/e14_preregistration.md และ e17a_preregistration.md

    python experiments/38_e14_cprime.py --one    # วัดเวลา 1 บล็อก
    python experiments/38_e14_cprime.py
→ outputs/experiments/E14/cprime_*.csv, results/e14/cprime_*.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import sys
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import tick_size
from forecast.evaluation.runner import load_or_build_dataset
from forecast.evaluation.splits import holdout_split, walk_forward_splits
from forecast.models.e9 import target_rows
from forecast.models.finetune_lr import KAPPAS, MIN_CV_DAYS, MIN_DAYS, Scaler, choose_c, fit_pretrain, inner_blocks, logloss

warnings.filterwarnings("ignore")


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


KEYS = ["date", "symbol", "session"]
TICK = ["fl_exp_ticks", "fl_prev_range_ticks", "f_tick_rel"]
FIXED_KAPPA = 3.0        # สเปกนี้: แถว fine-tune < 60 วัน
N_BAGS, BLOCK = 20, 20
OUT, RES = OUTPUT_DIR / "experiments" / "E14", ROOT / "results" / "e14"
SYM, PM = "d_sym_advanc", "d_sess_pm"


def add_tick_features(ds):
    """f_tick_rel = tick_size(P(c)) / P(c) × 100 · ใช้แค่ราคา ณ จุดตัด · k1 = แถว KBANK ช่อง 1 บาท"""
    d = ds.copy()
    tick = tick_size(d["p_ref"].to_numpy(float))
    d["f_tick_rel"] = tick / d["p_ref"].to_numpy(float) * 100
    d["tick1"] = tick >= 1
    d["k1"] = d["tick1"] & (d["symbol"] == "KBANK")
    d[SYM] = (d["symbol"] == "ADVANC").astype(float)
    d[PM] = (d["session"] == "afternoon").astype(float)
    return d


def fit_offset(Z, y, free, beta0, a0, lam):
    """loss = Σ log loss + λ‖β_free − β₀_free‖² · coefficient อื่นตรึงเป็น offset · intercept อิสระ → (β, a)"""
    if not np.isfinite(lam) or len(np.unique(y)) < 2:
        return beta0.copy(), a0
    off = Z @ beta0
    F = Z[:, free]

    def f(w):
        b, a = w[:-1], w[-1]
        s = off + F @ b + a
        p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        loss = np.logaddexp(0, -s)[y == 1].sum() + np.logaddexp(0, s)[y == 0].sum() + lam * b @ b
        g = p - y
        return loss, np.r_[F.T @ g + 2 * lam * b, g.sum()]

    w = minimize(f, np.r_[np.zeros(len(free)), a0], jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    beta = beta0.copy()
    beta[free] += w[:-1]
    return beta, float(w[-1])


def sigmoid(s):
    return 1 / (1 + np.exp(-np.clip(s, -30, 30)))


def pretrain(pre, cols, y, C=None):
    X = pre[cols].to_numpy(float)
    s = Scaler(X)
    Z = s(X)
    C = C or choose_c(Z, y, pre["date"].to_numpy())
    beta, a = fit_pretrain(Z, y, C)
    return s, beta, a, C


def start(s, beta, a, cols, drop):
    """ตัด dummy ที่คงที่ในชุด fine-tune: β = 0 แล้วรวมค่าเข้า intercept เริ่มต้น"""
    b, a0 = beta.copy(), a
    for c, v in drop.items():
        i = cols.index(c)
        a0 += b[i] * (v - s.mean[i]) / s.std[i]
        b[i] = 0.0
    return b, a0


def fine_tune(pre, yp, ft, yf, cols, free_names, drop, seed=0):
    """→ dict(pretrain, models, kappa, inner_ll, a0, a_mean, C) · ft = แถว fine-tune ก่อนวันทดสอบ"""
    s, beta, a, C = pretrain(pre, cols, yp)
    out = {"scaler": s, "beta": beta, "a": a, "C": C, "kappa": np.inf, "inner_ll": np.nan, "a0": np.nan, "a_mean": np.nan, "models": None}
    free = [cols.index(c) for c in free_names]
    dates = ft["date"].to_numpy()
    n = len(np.unique(dates))
    if n < MIN_DAYS:
        return out
    Zf = s(ft[cols].to_numpy(float))
    beta0, a0 = start(s, beta, a, cols, drop)
    kappa, ill = FIXED_KAPPA, np.nan
    if n >= MIN_CV_DAYS:
        scores = {k: [] for k in KAPPAS}
        for tr, va in inner_blocks(dates):
            cut = np.min(va)
            pp = pre["date"].to_numpy() < cut
            if len(np.unique(yp[pp])) < 2:
                continue
            s2, b2, a2, _ = pretrain(pre[pp], cols, yp[pp], C)
            b20, a20 = start(s2, b2, a2, cols, drop)
            A, B = dates < cut, np.isin(dates, va)
            Z2 = s2(ft[cols].to_numpy(float))
            for k in KAPPAS:
                bb, aa = fit_offset(Z2[A], yf[A], free, b20, a20, k / (2 * C))
                scores[k].append(logloss(yf[B], sigmoid(Z2[B] @ bb + aa)).mean())
        mean = {k: np.mean(v) for k, v in scores.items() if v}
        if mean:
            kappa = min(mean, key=mean.get)
            ill = mean[kappa]
    lam = kappa / (2 * C)
    models = []
    if np.isfinite(kappa):
        ud = np.unique(dates)
        blocks = [ud[i:i + BLOCK] for i in range(0, len(ud), BLOCK)]
        rng = np.random.default_rng(seed)
        for _ in range(N_BAGS):
            pick = np.concatenate([np.flatnonzero(np.isin(dates, blocks[j])) for j in rng.integers(0, len(blocks), len(blocks))])
            if len(np.unique(yf[pick])) == 2:
                models.append(fit_offset(Zf[pick], yf[pick], free, beta0, a0, lam))
    out.update(kappa=kappa, inner_ll=ill, a0=a0, models=models or [(beta0, a0)], a_mean=float(np.mean([m[1] for m in models])) if models else a0)
    return out


def predict(r, X, cols, tuned):
    """tuned = True → โมเดล fine-tune (ถ้ามี) · ไม่งั้น β₀"""
    Z = r["scaler"](X[cols].to_numpy(float))
    if tuned and r["models"]:
        return np.mean([sigmoid(Z @ b + a) for b, a in r["models"]], axis=0)
    return sigmoid(Z @ r["beta"] + r["a"])


def dir_block(d, fd, session, te, tag):
    start_day = np.min(te)
    tr = d[(d["date"].to_numpy() < start_day) & (d["session"] == session) & d["direction"].notna()]
    test = d[(d["session"] == session) & d["date"].isin(te)]
    base = [c for c in fd[session] if c not in TICK]
    cols = base + TICK + [SYM]
    y = (tr["direction"] == 1).to_numpy(int)
    K = tr[tr["k1"]]
    yk = (K["direction"] == 1).to_numpy(int)
    drop = {SYM: 0.0}
    c1 = fine_tune(tr, y, K, yk, cols, TICK, drop)
    out = test[KEYS + ["direction", "k1", "tick1", "p_ref"]].copy()
    out["p_pre"] = predict(c1, test, cols, False)
    out["p_cprime"] = out["p_pre"]
    m = test["k1"].to_numpy()
    if m.any():
        out.loc[m, "p_cprime"] = predict(c1, test[m], cols, True)
    # C-adv
    n_k = K["date"].nunique()
    fb, q10, q90, n_d = "", np.nan, np.nan, 0
    ca = None
    if n_k < MIN_DAYS:
        fb = "K<40"
    else:
        q10, q90 = K["fl_exp_ticks"].quantile([.1, .9])
        D = tr[(tr["symbol"] == "ADVANC") & (tr["fl_exp_ticks"] >= q10) & (tr["fl_exp_ticks"] <= q90)]
        n_d = len(D)
        if n_d < 200:
            fb = "donor<200"
        else:
            pool = pd.concat([K, D]).sort_values("date")
            yp = (pool["direction"] == 1).to_numpy(int)
            free = [c for c in cols if c != SYM]
            ca = fine_tune(pool, yp, K, yk, cols, free, drop)
    out["p_cadv"] = np.nan
    if m.any():
        out.loc[m, "p_cadv"] = predict(ca, test[m], cols, True) if ca else out.loc[m, "p_cprime"]
    out = out.assign(split=tag, kappa_c=c1["kappa"], inner_c=c1["inner_ll"], a0_c=c1["a0"], a_c=c1["a_mean"], kappa_a=ca["kappa"] if ca else np.nan,
                     inner_a=ca["inner_ll"] if ca else c1["inner_ll"], fallback=fb, n_k=n_k, n_d=n_d, q10=q10, q90=q90, block=str(start_day)[:10])
    return out


def price_block(d, eq_cols, up_cols, session, te):
    start_day = np.min(te)
    past = d[d["date"].to_numpy() < start_day]
    test = d[(d["session"] == session) & d["date"].isin(te)]
    out = test[KEYS].copy()
    out["block"] = str(start_day)[:10]
    m = test["k1"].to_numpy()
    res = {}
    for name, target, cols, pre_rows, drop in [("eq", "flat_price", eq_cols + [SYM, PM], past, {SYM: 0.0, PM: float(session == "afternoon")}),
                                               ("up", "price_updown", up_cols[session] + [SYM], past[past["session"] == session], {SYM: 0.0})]:
        a, ya = target_rows(pre_rows, target)
        pre = pre_rows[a]
        ses = pre[(pre["session"] == session) & pre["k1"]]
        b, yb = target_rows(ses, target) if target == "flat_price" else (np.ones(len(ses), bool), (ses["p_end"].to_numpy(float) > ses["p_ref"].to_numpy(float)).astype(int))
        r = fine_tune(pre, ya, ses[b], yb, cols, TICK, drop)
        out[f"{name}_pre"] = predict(r, test, cols, False)
        out[name] = out[f"{name}_pre"]
        if m.any():
            out.loc[m, name] = predict(r, test[m], cols, True)
        out[f"inner_{name}"], out[f"kappa_{name}"] = r["inner_ll"], r["kappa"]
    return out


def jobs(d, rules, holdout=False):
    ev = rules["evaluation"]
    out = []
    for session in ["morning", "afternoon"]:
        dates = d.loc[d["session"] == session, "date"].to_numpy()
        sp = holdout_split(dates, ev["holdout_days"]) if holdout else walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])
        out += [(session, te) for _, te in sp]
    return out


def b1_table(d):
    a = d[d["direction"].notna() & d["p_end"].notna()].copy()
    a["tick"] = tick_size(a["p_ref"].to_numpy(float))
    a["move"] = (a["p_end"] - a["p_start"]).abs() / a["tick"]
    a["pct"] = a["tick"] / a["p_ref"] * 100
    a["กลุ่ม"] = np.where(a["symbol"] == "ADVANC", np.where(a["tick"] >= 1, "ADVANC ช่อง 1 บาท", "ADVANC ช่องอื่น"), np.where(a["tick"] >= 1, "KBANK ช่อง 1 บาท", "KBANK ช่อง 0.5 บาท"))
    return a.groupby(["session", "กลุ่ม"]).agg(n=("move", "size"), ช่องต่อราคา=("pct", "mean"), ขยับมัธยฐาน=("move", "median"), วันนิ่ง=("direction", lambda s: (s == 0).mean()), วันขึ้น=("direction", lambda s: (s == 1).mean())).reset_index()


def main():
    s17, s18, s30, s31, s32 = [_module(n) for n in ["17_label_variants.py", "18_score_round4.py", "30_e16_up_notup.py", "31_e14_finetune.py", "32_e17a_price.py"]]
    rules = load_rules()
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    days = np.sort(ds["date"].unique())
    assert str(days[-1])[:10] == "2026-10-29" and str(days[-10])[:10] == "2026-10-14" and str(days[-11])[:10] == "2026-10-12"
    d = add_tick_features(ds)
    exps = load_experiments()
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")["features"]
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    eq_cols = list(v["aprice_lr"]["features"]) + ["f_tick_rel"]
    up_cols = {s: list(v["pprice_lr"]["features"][s]) + TICK for s in ("morning", "afternoon")}
    jb = jobs(d, rules)
    if "--one" in sys.argv:
        t = time.perf_counter()
        r = dir_block(d, fd, *jb[-1], "wf")
        print(f"ทิศทาง 1 บล็อก {time.perf_counter() - t:.1f} วิ · n_k {r['n_k'].iat[0]} · n_d {r['n_d'].iat[0]} · κ {r['kappa_c'].iat[0]} / {r['kappa_a'].iat[0]} · fallback '{r['fallback'].iat[0]}'")
        t = time.perf_counter()
        price_block(d, eq_cols, up_cols, *jb[-1])
        print(f"ราคา 1 บล็อก {time.perf_counter() - t:.1f} วิ · จำนวนบล็อก {len(jb)}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    b1 = b1_table(d)
    b1.to_csv(RES / "cprime_b1_table.csv", index=False, encoding="utf-8-sig")
    pr = pd.concat(Parallel(n_jobs=-1)(delayed(dir_block)(d, fd, s, te, "wf") for s, te in jb), ignore_index=True)
    pr.to_csv(OUT / "cprime_predictions.csv", index=False)
    ho = pd.concat(Parallel(n_jobs=-1)(delayed(dir_block)(d, fd, s, te, "holdout") for s, te in jobs(d, rules, True)), ignore_index=True)
    ho.to_csv(OUT / "cprime_holdout.csv", index=False)
    pp = pd.concat(Parallel(n_jobs=-1)(delayed(price_block)(d, eq_cols, up_cols, s, te) for s, te in jb), ignore_index=True)
    pp.to_csv(OUT / "prcprime_predictions.csv", index=False)
    print("ทิศทาง", len(pr), "· ราคา", len(pp), flush=True)
    assert str(pr["date"].max())[:10] == "2026-10-12"

    a_un = pd.read_csv(OUTPUT_DIR / "experiments" / "E16" / "a_un_predictions.csv")
    a_un["date"] = pd.to_datetime(a_un["date"], utc=True).dt.tz_convert(tz)
    a_ud, e0 = s18.load("C1235", "E12_fd_lr", tz), s30.e0_window(ds, rules)
    x = pr[KEYS + ["direction", "k1", "tick1", "p_pre", "p_cprime", "p_cadv"]].rename(columns={"p_pre": "p_C′-pretrain", "p_cprime": "p_C′", "p_cadv": "p_C-adv"})
    for n, p in [("A-un", a_un), ("A-ud", a_ud), ("E0-un", e0)]:
        x = x.merge(p[KEYS + ["p_up"]].rename(columns={"p_up": f"p_{n}"}), on=KEYS)
    x["p_C-adv"] = x["p_C-adv"].fillna(x["p_A-un"])
    x = x[x["direction"].notna()].reset_index(drop=True)
    n_boot, conf = rules["evaluation"]["n_boot"], 0.9875
    k1, tick1 = x["k1"].to_numpy(), x["tick1"].to_numpy()
    t = []
    for a in ["C′", "C-adv", "C′-pretrain"]:
        for b in ["A-un", "E0-un", "A-ud"]:
            t.append(s31.compare(x, a, b, conf, n_boot).assign(ช่วง="ทั้งช่วง"))
            t.append(s31.compare(x, a, b, conf, n_boot, k1).assign(ช่วง="KBANK ช่อง 1 บาท"))
            t.append(s31.compare(x, a, b, conf, n_boot, tick1).assign(ช่วง="ทุกแถวช่อง 1 บาท"))
    t = pd.concat(t, ignore_index=True)
    t.to_csv(RES / "cprime_compare.csv", index=False, encoding="utf-8-sig")
    get = lambda a, s, g: t[(t["a"] == a) & (t["b"] == "A-un") & (t["ช่วง"] == s) & (t["group"] == g)].iloc[0]
    ver = []
    for g, kg in [("เช้า", "KBANK เช้า"), ("บ่าย", "KBANK บ่าย")]:
        for a in ["C′", "C-adv"]:
            w, o = get(a, "KBANK ช่อง 1 บาท", kg), get(a, "ทั้งช่วง", g)
            ok = bool(w["d_ll_hi"] < 0 and w["d_acc_hi"] >= 0 and o["d_ll_lo"] <= 0 and o["d_acc_hi"] >= 0)
            ver.append({"รอบ": g, "แบบ": a, "n": w["n"], "d_ll": w["d_ll"], "lo": w["d_ll_lo"], "hi": w["d_ll_hi"], "d_acc": w["d_acc"], "acc_lo": w["d_acc_lo"], "acc_hi": w["d_acc_hi"],
                        "ทั้งช่วง_d_ll_lo": o["d_ll_lo"], "ทั้งช่วง_d_acc_hi": o["d_acc_hi"], "ผ่าน": ok})
    ver = pd.DataFrame(ver)
    ver.to_csv(RES / "cprime_verdict.csv", index=False, encoding="utf-8-sig")
    # ตัวแทน: inner CV บนบล็อกร่วม · ไม่ดูผลชุดทดสอบ
    rep = []
    for sess in ["morning", "afternoon"]:
        p = pr[pr["session"] == sess].drop_duplicates("block")
        j = p[["inner_c", "inner_a"]].dropna()
        rep.append({"ช่อง": f"{sess}_dir", "บล็อกร่วม": len(j), "inner_C′": j["inner_c"].mean(), "inner_C-adv": j["inner_a"].mean(), "ตัวแทน": "C′" if j["inner_c"].mean() <= j["inner_a"].mean() else "C-adv",
                    "κ_C′": p["kappa_c"].value_counts().to_dict(), "κ_C-adv": p["kappa_a"].value_counts().to_dict(), "fallback": p["fallback"].value_counts().to_dict(),
                    "intercept_β₀": p["a0_c"].dropna().iloc[-1] if p["a0_c"].notna().any() else np.nan, "intercept_ft": p["a_c"].dropna().iloc[-1] if p["a_c"].notna().any() else np.nan,
                    "n_k_สุดท้าย": p["n_k"].iloc[-1], "n_d_สุดท้าย": p["n_d"].iloc[-1], "q10": p["q10"].iloc[-1], "q90": p["q90"].iloc[-1]})
    rep = pd.DataFrame(rep)
    rep.to_csv(RES / "cprime_representatives.csv", index=False, encoding="utf-8-sig")

    # PR-C′
    eq, up, zq = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr", "E9_baseline_ud"]]
    QC = ["q10", "q25", "q50", "q75", "q90"]
    y = d[KEYS + ["p_ref", "p_end", "k1", "tick1"]].merge(pp, on=KEYS)
    y = y.merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(up[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS).merge(zq[KEYS + QC], on=KEYS)
    y = y[y["p_end"].notna() & y["p_ref"].notna()].reset_index(drop=True)
    pref = y["p_ref"].to_numpy(float)
    y["pt_PR-A"], y["pt_P(c)"] = s18.price_rule(pref, y["eq_A"].to_numpy(), y["up_A"].to_numpy())[0], pref
    y["pt_PR-C′"] = s18.price_rule(pref, y["eq"].to_numpy(), y["up"].to_numpy())[0]
    y["pt_PR-C′-pretrain"] = s18.price_rule(pref, y["eq_pre"].to_numpy(), y["up_pre"].to_numpy())[0]
    s32.CONF = conf
    names = ["PR-A", "P(c)", "PR-C′-pretrain", "PR-C′"]
    t2 = s32.score(y.assign(tick1=y["k1"]), names, n_boot)
    t2["ช่วง"] = t2["ช่วง"].replace({"ช่องราคา 1 บาท": "KBANK ช่อง 1 บาท"})
    t2.to_csv(RES / "prcprime_metrics.csv", index=False, encoding="utf-8-sig")
    cov = pd.concat([s32.coverage(y[y["k1"]].reset_index(drop=True), ["PR-A", "PR-C′"], y[y["k1"]].reset_index(drop=True)[QC], "KBANK ช่อง 1 บาท"), s32.coverage(y, ["PR-A", "PR-C′"], y[QC], "ทั้งช่วง")])
    cov.to_csv(RES / "prcprime_coverage.csv", index=False, encoding="utf-8-sig")
    gp = lambda n, s, g: t2[(t2["model"] == n) & (t2["ช่วง"] == s) & (t2["group"] == g)].iloc[0]
    cg = lambda n, g, c: cov[(cov["quantile"] == "KBANK ช่อง 1 บาท") & (cov["model"] == n) & (cov["group"] == g)][c].iat[0]
    pv = []
    for g, kg in [("เช้า", "KBANK เช้า"), ("บ่าย", "KBANK บ่าย")]:
        w, o = gp("PR-C′", "KBANK ช่อง 1 บาท", kg), gp("PR-C′", "ทั้งช่วง", g)
        win = (w["d_mae_hi"] < 0 and w["d_rmse_lo"] <= 0) or (w["d_rmse_hi"] < 0 and w["d_mae_lo"] <= 0)
        other = o["d_mae_lo"] <= 0 and o["d_rmse_lo"] <= 0
        cv_ok = abs(cg("PR-C′", kg, "cover50") - .5) <= abs(cg("PR-A", kg, "cover50") - .5) and abs(cg("PR-C′", kg, "cover80") - .8) <= abs(cg("PR-A", kg, "cover80") - .8)
        pv.append({"รอบ": g, "n": w["n"], "mae": w["mae"], "d_mae": w["d_mae"], "mae_lo": w["d_mae_lo"], "mae_hi": w["d_mae_hi"], "rmse": w["rmse"], "d_rmse": w["d_rmse"], "rmse_lo": w["d_rmse_lo"], "rmse_hi": w["d_rmse_hi"],
                   "ชนะ": bool(win), "ทั้งช่วงไม่แย่": bool(other), "coverage": bool(cv_ok), "ผ่าน": bool(win and other and cv_ok)})
    pv = pd.DataFrame(pv)
    pv.to_csv(RES / "prcprime_verdict.csv", index=False, encoding="utf-8-sig")
    h = ho[ho["direction"].notna()]
    yh = (h["direction"] == 1).to_numpy(float)
    hr = []
    for n in ["p_pre", "p_cprime"]:
        for s in ["morning", "afternoon"]:
            m = h["session"] == s
            hr.append({"model": n, "session": s, "n": int(m.sum()), "acc": ((h.loc[m, n] > .5) == (yh[m] == 1)).mean(), "ll": logloss(yh[m], h.loc[m, n].to_numpy()).mean()})
    pd.DataFrame(hr).to_csv(RES / "cprime_holdout.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    print("\nB1\n", b1.round(3).to_string(index=False))
    show = ["a", "b", "ช่วง", "group", "n", "acc_a", "acc_b", "pred_up_a", "d_ll", "d_ll_lo", "d_ll_hi", "d_acc", "d_acc_lo", "d_acc_hi"]
    print(t[t["group"].isin(["เช้า", "บ่าย", "KBANK เช้า", "KBANK บ่าย"]) & (t["b"].isin(["A-un", "E0-un"])) & (t["ช่วง"] != "ทุกแถวช่อง 1 บาท")][show].round(4).to_string(index=False))
    print("\nคำตัดสิน C′ / C-adv\n", ver.round(4).to_string(index=False))
    print("\nตัวแทน\n", rep.round(4).T.to_string())
    print("\nPR-C′\n", pv.round(4).to_string(index=False))
    print("\nholdout\n", pd.DataFrame(hr).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
