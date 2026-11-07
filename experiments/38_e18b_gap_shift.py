"""E18b: เลื่อนราคาอ้างอิงรอบเช้าด้วยค่าทาย overnight gap — เกณฑ์ใน results/e18b_preregistration.md · ไม่แตะระบบ live

    python experiments/38_e18b_gap_shift.py a         # ขั้น A: ĝ นอกตัวอย่าง + จุดหยุดเทียบ H1
    python experiments/38_e18b_gap_shift.py time      # วัดเวลา 1 รอบฝึก ไม่คำนวณคะแนน
    python experiments/38_e18b_gap_shift.py full      # ขั้น A + B + รายงานตาราง
→ outputs/experiments/E18b/ (มีราคาจริง ไม่อยู่ใน git) · results/e18b/ (ตัวเลขสรุปเท่านั้น)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import importlib.util
import sys
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import round_to_tick, tick_size
from forecast.evaluation.runner import load_or_build_dataset, run_variant
from forecast.evaluation.splits import walk_forward_splits
from forecast.features.overnight import build_features
from forecast.models.e9 import inner_folds

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
FX = ["fx_spx", "fx_thd_prem", "fx_asia_gap", "fx_usdthb"]
ALPHAS = (0.1, 1.0, 10.0, 100.0)
BLOCK, MIN_TRAIN = 5, 120
CONF = 1 - 0.05 / 3
H1_LOW = {"ADVANC": 0.252, "KBANK": 0.182}
H1_CORR = {"ADVANC": 0.345, "KBANK": 0.301}
MODELS = ["H2b-0", "H2b-rule", "H2b-mean"]
NAMES = ["PR-A", "P(c)"] + MODELS
QC = ["q10", "q25", "q50", "q75", "q90"]
RES, OUT = ROOT / "results" / "e18b", OUTPUT_DIR / "experiments" / "E18b"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------- ขั้น A ----------
def add_asia_gap(fx):
    """ค่าเฉลี่ยของตลาดเอเชียที่มีค่า · วันหยุดทั้งสองประเทศ = 0 (ไม่มีข่าวใหม่)"""
    out = fx.copy()
    out["fx_asia_gap"] = out[["fx_n225_gap", "fx_kospi_gap"]].mean(axis=1, skipna=True).fillna(0.0)
    return out


def morning_frame(ds, fx):
    """แถวเช้า + feature + gap → (DataFrame, จำนวนค่าว่างที่แทน 0)"""
    am = ds[ds["session"] == "morning"].merge(add_asia_gap(fx)[["date"] + FX], on="date", how="left")
    missing = am[FX].isna().sum().to_dict()
    am[FX] = am[FX].fillna(0.0)
    am["gap"] = np.log(am["p_start"] / am["p_ref"])
    return am.sort_values(["symbol", "date"]).reset_index(drop=True), missing


def _pipe(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def pick_alpha(X, y, dates):
    scores = {}
    for a in ALPHAS:
        mse = [np.mean((y[va] - _pipe(a).fit(X[tr], y[tr]).predict(X[va])) ** 2) for tr, va in inner_folds(dates)]
        scores[a] = np.mean(mse) if mse else np.inf
    return min(scores, key=scores.get)


def gap_oos(am, block=BLOCK, min_train=MIN_TRAIN):
    """ĝ นอกตัวอย่างทุกแถว: โมเดลฝึกจากวันก่อนบล็อกเท่านั้น → (แถว KEYS + g_hat + gap_model_train_end, สัมประสิทธิ์)"""
    rows, coefs = [], []
    for sym, g in am.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = g["date"].to_numpy()
        pred = np.zeros(len(g))
        end = pd.Series(pd.NaT, index=g.index, dtype=g["date"].dtype)
        y, X = g["gap"].to_numpy(float), g[FX].to_numpy(float)
        for _, te in walk_forward_splits(dates, block, min_train, 0):
            start = np.min(te)
            a, b = (dates < start) & np.isfinite(y), np.isin(dates, te)
            alpha = pick_alpha(X[a], y[a], dates[a])
            m = _pipe(alpha).fit(X[a], y[a])
            pred[b] = m.predict(X[b])
            end[b] = g.loc[a, "date"].max()
            coefs.append({"symbol": sym, "start": str(start)[:10], "alpha": alpha, **dict(zip(FX, m[-1].coef_ / m[0].scale_))})
        rows.append(g[KEYS].assign(g_hat=pred, gap_model_train_end=end))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coefs)


def assert_no_leak(gh):
    h = gh[gh["gap_model_train_end"].notna()]
    assert (h["gap_model_train_end"] < h["date"]).all(), "โมเดล gap ใช้ข้อมูลวันทาย"
    assert (gh.loc[gh["gap_model_train_end"].isna(), "g_hat"] == 0).all(), "แถวก่อนฝึกครบต้องเป็น ĝ = 0"


def boot_slope(x, y, dates, n=2000, seed=0):
    df = pd.DataFrame({"x": x, "y": y, "d": dates}).dropna()
    groups = [g[["x", "y"]].to_numpy() for _, g in df.groupby("d")]
    idx = np.random.default_rng(seed).integers(0, len(groups), (n, len(groups)))
    b = []
    for row in idx:
        s = np.concatenate([groups[i] for i in row])
        b.append(np.polyfit(s[:, 0], s[:, 1], 1)[0])
    return np.polyfit(df["x"], df["y"], 1)[0], np.quantile(b, .025), np.quantile(b, .975), len(df)


def wf_rows(g, rules):
    """แถวในช่วงทดสอบ walk-forward (ไม่รวม holdout)"""
    ev = rules["evaluation"]
    days = [d for _, te in walk_forward_splits(g["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]) for d in te]
    return g["date"].isin(days)


def stage_a(am, gh, coefs, missing, rules, boot_corr):
    x = am.merge(gh, on=KEYS)
    rows, rev = [], []
    for sym, g in x.groupby("symbol"):
        g = g[wf_rows(g, rules)]
        r, lo, hi, n = boot_corr(g["g_hat"].to_numpy(), g["gap"].to_numpy(), g["date"].to_numpy(), rules["evaluation"]["n_boot"])
        c = coefs[coefs["symbol"] == sym]
        rows.append({"symbol": sym, "n": n, "corr": r, "lo": lo, "hi": hi, "h1_corr": H1_CORR[sym], "sd_g_hat": g["g_hat"].std(), "sd_gap": g["gap"].std(),
                     "alpha_mode": c["alpha"].mode().iloc[0], **{f"coef_{f}": c[f].mean() for f in FX}})
        s, lo2, hi2, n2 = boot_slope(g["gap"].to_numpy(), np.log(g["p_end"] / g["p_start"]).to_numpy(), g["date"].to_numpy())
        rev.append({"symbol": sym, "n": n2, "slope": s, "lo": lo2, "hi": hi2})
    t, r = pd.DataFrame(rows), pd.DataFrame(rev)
    t.to_csv(RES / "stage_a.csv", index=False, encoding="utf-8-sig")
    r.to_csv(RES / "reversal.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([missing]).to_csv(RES / "missing_filled.csv", index=False, encoding="utf-8-sig")
    print("ค่าว่างที่แทน 0:", missing)
    print(t.round(4).to_string(index=False))
    print(r.round(4).to_string(index=False), flush=True)
    return bool(all(t.set_index("symbol").loc[s, "corr"] >= H1_LOW[s] for s in H1_LOW))


# ---------- ขั้น B ----------
def shifted(ds, gh):
    """ds + g_hat, g_abs, pc (P(c) เดิม) และ p_ref ของแถวเช้าเป็น P(c)′ · feature เดิมทุกคอลัมน์ · แถวบ่าย ĝ = 0"""
    d = ds.merge(gh[KEYS + ["g_hat"]], on=KEYS, how="left")
    d["g_hat"] = d["g_hat"].fillna(0.0)
    d["g_abs"] = d["g_hat"].abs()
    d["pc"] = d["p_ref"]
    am = (d["session"] == "morning").to_numpy()
    d.loc[am, "p_ref"] = np.asarray(round_to_tick(d.loc[am, "pc"].to_numpy(float) * np.exp(d.loc[am, "g_hat"].to_numpy(float))), float)
    return d


def variants(exps):
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    eq = {**copy.deepcopy(v["aprice_lr"]), "features": list(v["aprice_lr"]["features"]) + ["g_hat", "g_abs"]}
    f = v["pprice_lr"]["features"]
    up = {**copy.deepcopy(v["pprice_lr"]), "sessions": ["morning"], "features": {"morning": list(f["morning"]) + ["g_hat", "g_abs"], "afternoon": list(f["afternoon"])}}
    base_up = {**copy.deepcopy(v["pprice_lr"]), "sessions": ["morning"]}
    return eq, up, copy.deepcopy(v["aprice_lr"]), base_up


def cond_means(am):
    """E[z′|สูงกว่า], E[z′|ต่ำกว่า] จากวันก่อนวันทาย แยกหุ้น"""
    h = am.sort_values(["symbol", "date"]).copy()
    z = np.log(h["p_end"].astype(float) / h["p_ref"].astype(float))
    h["z_up"] = z.where(z > 0).groupby(h["symbol"]).transform(lambda s: s.expanding().mean().shift(1))
    h["z_dn"] = z.where(z < 0).groupby(h["symbol"]).transform(lambda s: s.expanding().mean().shift(1))
    return h[KEYS + ["z_up", "z_dn"]]


def mean_point(p_ref, p_eq, p_price, z_up, z_dn):
    return p_ref * np.exp((1 - p_eq) * p_price * z_up + (1 - p_eq) * (1 - p_price) * z_dn)


def band_quantiles(am, blocks, quantiles):
    """quantile ของ z′ จากวันก่อนบล็อก แยกหุ้น หักค่ามัธยฐาน → แถว KEYS + zs_*"""
    z = np.log(am["p_end"].astype(float) / am["p_ref"].astype(float))
    out = []
    for te in blocks:
        start = np.min(te)
        for sym, g in am[am["date"].isin(te)].groupby("symbol"):
            h = z[(am["symbol"] == sym) & (am["date"] < start)].dropna()
            q = np.quantile(h, quantiles) - np.median(h)
            out.append(g[KEYS].assign(**{f"zs_{c}": v for c, v in zip(QC, q)}))
    return pd.concat(out, ignore_index=True)


def boot_rmse(ea, eb, dates, conf, n_boot, seed=0):
    """RMSE(a) − RMSE(b): คำนวณ RMSE ใหม่ทุกรอบจากวันที่สุ่มได้ จับคู่วันเดียวกัน"""
    df = pd.DataFrame({"a": np.asarray(ea, float) ** 2, "b": np.asarray(eb, float) ** 2, "d": np.asarray(dates)}).groupby("d").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    a, b, n = df["a"].to_numpy(), df["b"].to_numpy(), df["n"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(df), (n_boot, len(df)))
    diff = np.sqrt(a[idx].sum(1) / n[idx].sum(1)) - np.sqrt(b[idx].sum(1) / n[idx].sum(1))
    al = (1 - conf) / 2
    return np.sqrt(a.sum() / n.sum()) - np.sqrt(b.sum() / n.sum()), np.quantile(diff, al), np.quantile(diff, 1 - al)


def assemble(d_shift, ds, p_eq, p_up, base_eq, base_up, base_q, blocks, rules, s18):
    """แถวเช้าที่มีผลครบ + ราคาจุดทุกแบบ + ช่วงราคา"""
    am = d_shift[d_shift["session"] == "morning"].copy()
    x = am[KEYS + ["pc", "p_ref", "p_end", "p_start", "g_hat"]].rename(columns={"p_ref": "pp"})
    for name, p in [("eq_B", p_eq), ("up_B", p_up), ("eq_A", base_eq), ("up_A", base_up)]:
        x = x.merge(p[KEYS + ["p_up"]].rename(columns={"p_up": name}), on=KEYS)
    x = x.merge(base_q[KEYS + QC], on=KEYS).merge(cond_means(am), on=KEYS).merge(band_quantiles(am, blocks, rules["quantiles"]), on=KEYS)
    x = x[x["p_end"].notna() & x["pc"].notna()].reset_index(drop=True)
    pc, pp = x["pc"].to_numpy(float), x["pp"].to_numpy(float)
    x["pt_PR-A"] = s18.price_rule(pc, x["eq_A"].to_numpy(), x["up_A"].to_numpy())[0]
    x["pt_P(c)"] = pc
    x["pt_H2b-0"] = pp
    x["pt_H2b-rule"] = s18.price_rule(pp, x["eq_B"].to_numpy(), x["up_B"].to_numpy())[0]
    x["pt_H2b-mean"] = mean_point(pp, x["eq_B"].to_numpy(), x["up_B"].to_numpy(), x["z_up"].to_numpy(), x["z_dn"].to_numpy())
    assert x[[f"pt_{n}" for n in NAMES]].notna().all().all()
    return x


def groups(x):
    return [("เช้า", np.ones(len(x), bool)), ("KBANK เช้า", (x["symbol"] == "KBANK").to_numpy()), ("ADVANC เช้า", (x["symbol"] == "ADVANC").to_numpy())]


def score(x, n_boot, s31, conf=CONF):
    pe, dates, rows = x["p_end"].to_numpy(float), x["date"].to_numpy(), []
    for name in NAMES:
        e = pe - x[f"pt_{name}"].to_numpy(float)
        for lab, m in groups(x):
            if not m.any():
                continue
            r = {"model": name, "group": lab, "n": int(m.sum()), "n_days": len(np.unique(dates[m])), "rmse": np.sqrt((e[m] ** 2).mean()), "mae": np.abs(e[m]).mean()}
            for base, tag in [("PR-A", "pra"), ("P(c)", "pc")]:
                if name == base:
                    continue
                b = pe - x[f"pt_{base}"].to_numpy(float)
                r.update(zip([f"d_rmse_{tag}", f"d_rmse_{tag}_lo", f"d_rmse_{tag}_hi"], boot_rmse(e[m], b[m], dates[m], conf, n_boot)))
                r.update(zip([f"d_mae_{tag}", f"d_mae_{tag}_lo", f"d_mae_{tag}_hi"], s31.boot_ci(np.abs(e[m]) - np.abs(b[m]), dates[m], conf, n_boot)))
            rows.append(r)
    return pd.DataFrame(rows)


def coverage(x):
    pe, rows = x["p_end"].to_numpy(float), []
    for name in ["PR-A"] + MODELS:
        pt = x[f"pt_{name}"].to_numpy(float)
        rnd = (lambda v: np.round(v, 2)) if name == "H2b-mean" else (lambda v: np.asarray(round_to_tick(v), float))
        pre = "" if name == "PR-A" else "zs_"
        band = {c: rnd(pt * np.exp(x[pre + c].to_numpy(float) - x[pre + "q50"].to_numpy(float))) for c in QC}
        in50, in80 = (pe >= band["q25"]) & (pe <= band["q75"]), (pe >= band["q10"]) & (pe <= band["q90"])
        for lab, m in groups(x):
            rows.append({"model": name, "group": lab, "cover50": in50[m].mean(), "cover80": in80[m].mean(), "width50": (band["q75"] - band["q25"])[m].mean(), "width80": (band["q90"] - band["q10"])[m].mean()})
    return pd.DataFrame(rows)


def verdict(t, cov):
    q, c = t.set_index(["model", "group"]), cov.set_index(["model", "group"])
    base = c.loc[("PR-A", "เช้า")]
    rows = []
    for m in MODELS:
        a = q.loc[(m, "เช้า")]
        k1 = bool(a["d_rmse_pra_hi"] < 0)
        k2 = all(q.loc[(m, f"{s} เช้า"), "d_rmse_pra_lo"] <= 0 for s in ("KBANK", "ADVANC"))
        k3 = bool(a["d_mae_pra_lo"] <= 0)
        k4 = all(abs(c.loc[(m, "เช้า"), f"cover{lv}"] - lv / 100) <= abs(base[f"cover{lv}"] - lv / 100) for lv in (50, 80))
        rows.append({"model": m, "c1_rmse": k1, "c2_stocks": k2, "c3_mae": k3, "c4_coverage": k4, "pass": k1 and k2 and k3 and k4})
    return pd.DataFrame(rows)


def inner_cv(d_shift, eq, up, rules, s18):
    """RMSE เฉลี่ยของ 3 บล็อกท้ายในข้อมูลฝึกครั้งสุดท้าย — ใช้เมื่อผ่านมากกว่าหนึ่งแบบเท่านั้น"""
    from forecast.evaluation.runner import design_matrix, select_features
    from forecast.models import get_model
    ev = rules["evaluation"]
    am = d_shift[d_shift["session"] == "morning"]
    last = list(walk_forward_splits(am["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]))[-1][0]
    h = am[am["date"].isin(last)]
    means = cond_means(am)
    scores = {m: [] for m in MODELS}
    for tr_m, va_m in inner_folds(h["date"].to_numpy()):
        tr_days, va_days = np.unique(h["date"].to_numpy()[tr_m]), np.unique(h["date"].to_numpy()[va_m])
        def fit(var, g):
            X = design_matrix(g, select_features(g, var), var.get("group_by", []))
            dates, lab = g["date"].to_numpy(), g["up"].notna().to_numpy()
            tr, te = np.isin(dates, tr_days) & lab, np.isin(dates, va_days)
            m = get_model(var["model"], rules["quantiles"], **var.get("params", {})).fit(X[tr], g["up"].to_numpy()[tr], g["z"].to_numpy()[tr], g[tr])
            return g.loc[te, KEYS].assign(p_up=m.predict(X[te], g[te])["p_up"].to_numpy())
        va = fit(eq, d_shift).rename(columns={"p_up": "eq"}).merge(fit(up, am).rename(columns={"p_up": "up"}), on=KEYS)
        va = va.merge(am[KEYS + ["p_ref", "p_end"]], on=KEYS).merge(means, on=KEYS).dropna(subset=["p_end", "p_ref"])
        pp, pe = va["p_ref"].to_numpy(float), va["p_end"].to_numpy(float)
        pts = {"H2b-0": pp, "H2b-rule": s18.price_rule(pp, va["eq"].to_numpy(), va["up"].to_numpy())[0], "H2b-mean": mean_point(pp, va["eq"].to_numpy(), va["up"].to_numpy(), va["z_up"].to_numpy(), va["z_dn"].to_numpy())}
        for m, p in pts.items():
            scores[m].append(np.sqrt(np.mean((pe - p) ** 2)))
    return {m: float(np.mean(v)) for m, v in scores.items()}


def moved_table(x):
    pc, pp, ps, pe = [x[c].to_numpy(float) for c in ("pc", "pp", "p_start", "p_end")]
    moved = pp != pc
    ok = np.isfinite(ps)
    right = (np.sign(pp - pc) == np.sign(ps - pc))
    rows = []
    for lab, m in groups(x):
        r = {"group": lab, "n": int(m.sum()), "share_moved": moved[m].mean(), "moved_right_vs_gap": right[m & moved & ok].mean() if (m & moved & ok).any() else np.nan}
        for name in NAMES:
            e = pe - x[f"pt_{name}"].to_numpy(float)
            r[f"rmse_moved_{name}"] = np.sqrt((e[m & moved] ** 2).mean()) if (m & moved).any() else np.nan
            r[f"rmse_unmoved_{name}"] = np.sqrt((e[m & ~moved] ** 2).mean()) if (m & ~moved).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def decomposition(x, best):
    h = x[x["p_start"].notna()]
    rows = []
    for name in ["PR-A", best]:
        pt = h[f"pt_{name}"].to_numpy(float)
        for lab, m in groups(h):
            a, b = h["p_start"].to_numpy(float)[m] - pt[m], (h["p_end"] - h["p_start"]).to_numpy(float)[m]
            rows.append({"model": name, "group": lab, "n": int(m.sum()), "rmse_gap_part": np.sqrt((a ** 2).mean()), "rmse_after_open": np.sqrt((b ** 2).mean()), "rmse_total": np.sqrt(((a + b) ** 2).mean())})
    return pd.DataFrame(rows)


def report_window(ds, days, gh, exps, rules, s18, zq_var):
    """holdout / ข้อมูลใหม่: โมเดลราคาฝึกก่อนช่วงนี้ · ĝ จากโมเดลที่ฝึกก่อนวันนั้น · รายงานเท่านั้น"""
    r = copy.deepcopy(rules)
    r["evaluation"]["holdout_days"] = days
    d = shifted(ds, gh)
    eq, up, b_eq, b_up = variants(exps)
    p_eq, p_up = run_variant(d, eq, r, holdout_eval=True), run_variant(d, up, r, holdout_eval=True)
    q_eq, q_up = run_variant(ds, b_eq, r, holdout_eval=True), run_variant(ds, b_up, r, holdout_eval=True)
    base_q = run_variant(ds[ds["session"] == "morning"], zq_var, r, holdout_eval=True)
    block = [np.sort(ds["date"].unique())[-days:]]
    return assemble(d, ds, p_eq, p_up, q_eq, q_up, base_q, block, rules, s18)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    t0 = time.perf_counter()
    s17, s18, s31, s37 = [_module(n) for n in ["17_label_variants.py", "18_score_round4.py", "31_e14_finetune.py", "37_e18_overnight.py"]]
    rules = load_rules()
    assert rules["label"]["p_ref_morning"] == "official_close" and rules["label"]["price_1630"] == "closing_auction" and rules["sessions"]["afternoon"]["cutoff"] == "12:35"
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    ud = np.sort(ds["date"].unique())
    assert str(ud[-11])[:10] == "2026-10-12" and str(ud[-10])[:10] == "2026-10-14" and str(ud[-1])[:10] == "2026-10-29", (ud[-11], ud[-10], ud[-1])
    tables, hourly = s37.load_tables()
    fx = build_features(pd.DatetimeIndex(ud), tables, hourly)
    RES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    am, missing = morning_frame(ds, fx)
    gh, coefs = gap_oos(am)
    assert_no_leak(gh)
    gh.to_csv(OUT / "g_hat.csv", index=False)
    coefs.to_csv(RES / "gap_coefs.csv", index=False, encoding="utf-8-sig")
    print(f"ขั้น A ฝึกเสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)
    exps = load_experiments()
    eq, up, b_eq, b_up = variants(exps)
    d = shifted(ds, gh)
    if mode == "time":
        t1 = time.perf_counter()
        run_variant(d, eq, rules, holdout_eval=True)
        print(f"A-price′ 1 รอบฝึก {time.perf_counter() - t1:.1f} วิ", flush=True)
        t1 = time.perf_counter()
        run_variant(d, up, rules, holdout_eval=True)
        print(f"p_price′ 1 รอบฝึก {time.perf_counter() - t1:.1f} วิ", flush=True)
        return
    ok = stage_a(am, gh, coefs, missing, rules, s37.boot_corr)
    if mode == "a":
        print("จุดหยุด:", "ผ่าน" if ok else "correlation ต่ำกว่าขอบล่าง H1 → หาบั๊กก่อน")
        return
    if not ok:
        print("หยุด: correlation ต่ำกว่าขอบล่าง H1 → หาบั๊กก่อนไปขั้น B")
        return

    p_eq, p_up = run_variant(d, eq, rules), run_variant(d, up, rules)
    p_eq.to_csv(OUT / "aprice_predictions.csv", index=False)
    p_up.to_csv(OUT / "pprice_predictions.csv", index=False)
    print(f"ขั้น B ฝึกเสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)
    base_eq, base_up, base_q = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr", "E9_baseline_ud"]]
    ev = rules["evaluation"]
    blocks = [te for _, te in walk_forward_splits(am["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
    x = assemble(d, ds, p_eq, p_up, base_eq, base_up, base_q, blocks, rules, s18)
    x.to_csv(OUT / "points.csv", index=False)
    t, cov = score(x, n_boot, s31), coverage(x)
    v = verdict(t, cov)
    passed = v.loc[v["pass"], "model"].tolist()
    chosen = passed[0] if len(passed) == 1 else None
    if len(passed) > 1:
        cv = inner_cv(d, eq, up, rules, s18)
        pd.DataFrame([cv]).to_csv(RES / "inner_cv.csv", index=False, encoding="utf-8-sig")
        chosen = min(passed, key=cv.get)
    v["chosen"] = v["model"] == chosen
    best = t[(t["group"] == "เช้า") & t["model"].isin(MODELS)].sort_values("rmse")["model"].iloc[0]
    tick = x[np.asarray(tick_size(x["pc"].to_numpy(float))) >= 1].reset_index(drop=True)
    for name, tab in [("metrics", t), ("coverage", cov), ("verdict", v), ("moved", moved_table(x)), ("decomposition", decomposition(x, best)), ("tick1", score(tick, n_boot, s31))]:
        tab.to_csv(RES / f"{name}.csv", index=False, encoding="utf-8-sig")

    zq_var = {"name": "b", "model": "baseline", "params": {"target": "updown"}}
    ho = report_window(ds, 10, gh, exps, rules, s18, zq_var)
    ho_t = score(ho, n_boot, s31)
    ho_t.to_csv(RES / "holdout.csv", index=False, encoding="utf-8-sig")
    new_ds = load_or_build_dataset(rules, tag="_new")
    nd = np.sort(new_ds["date"].unique())
    assert str(nd[-6])[:10] == "2026-10-30" and str(nd[-1])[:10] == "2026-11-06", nd[-7:]
    fx2 = build_features(pd.DatetimeIndex(nd), tables, hourly)
    am2, missing2 = morning_frame(new_ds, fx2)
    gh2, _ = gap_oos(am2)
    assert_no_leak(gh2)
    nw = report_window(new_ds, 6, gh2, exps, rules, s18, zq_var)
    nw_t = score(nw, n_boot, s31)
    nw_t.to_csv(RES / "new_data.csv", index=False, encoding="utf-8-sig")
    nw.to_csv(OUT / "new_rows.csv", index=False)
    pd.DataFrame([{"sec_total": time.perf_counter() - t0, "missing_new": str(missing2)}]).to_csv(RES / "runtime.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    show = ["model", "group", "n", "rmse", "mae", "d_rmse_pra", "d_rmse_pra_lo", "d_rmse_pra_hi", "d_mae_pra", "d_mae_pra_lo", "d_mae_pra_hi", "d_rmse_pc", "d_rmse_pc_lo", "d_rmse_pc_hi"]
    for tag, tab in [("ทั้งช่วง", t), ("ช่องราคา 1 บาท", pd.read_csv(RES / "tick1.csv")), ("holdout", ho_t), ("ข้อมูลใหม่", nw_t)]:
        print("---", tag)
        print(tab[show].round(3).to_string(index=False))
    print(cov.round(3).to_string(index=False))
    print(v.to_string(index=False))
    print(pd.read_csv(RES / "moved.csv").round(3).T.to_string())
    print(pd.read_csv(RES / "decomposition.csv").round(3).to_string(index=False))
    print(f"รวม {time.perf_counter() - t0:.0f} วิ", flush=True)


if __name__ == "__main__":
    main()
