"""E26: feature ใหม่ F1–F3 ทีละกลุ่ม — เกณฑ์ใน results/e26_preregistration.md · รายงานเท่านั้น

    python experiments/53_e26_features.py
→ outputs/experiments/E26/ (มีราคาจริง ไม่อยู่ใน git) · results/e26/ (ตัวเลขสรุปเท่านั้น)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import importlib.util
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.config import OUTPUT_DIR, RAW_DIR, ROOT, load_rules, session_times
from forecast.data.io import load_bars
from forecast.evaluation.runner import design_matrix, run_variant, select_features
from forecast.evaluation.splits import walk_forward_splits
from forecast.models import get_model

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
CONF = 1 - 0.05 / 3
F1 = ["fm_set50_1d"]
F2 = ["fo_on_1", "fo_on_mean5", "fo_on_mean20", "fo_id_1", "fo_id_mean5", "fo_id_mean20"]
F3 = ["fv_rv60_1", "fv_rv60_5", "fv_rv60_20"]
F3_PM = ["fv_rv60_am"]
GROUPS = {"F1": {"morning": F1, "afternoon": F1}, "F2": {"morning": F2, "afternoon": F2}, "F3": {"morning": F3, "afternoon": F3 + F3_PM}}
DIRECTION_GROUPS = ("F1", "F2")
OUT, RES = OUTPUT_DIR / "experiments" / "E26", ROOT / "results" / "e26"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def naive_days(dates):
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).astype(str).str[:10]))


def last_before(series, days):
    """ค่าล่าสุดของ series ที่วันที่ < แต่ละวันใน days · ไม่มี = NaN · series ต้องเรียงตามวันที่"""
    i = series.index.searchsorted(days, side="left") - 1
    out = np.full(len(days), np.nan)
    ok = i >= 0
    out[ok] = series.to_numpy(float)[i[ok]]
    return out


def daily_features(daily, days):
    """F2: แยกผลตอบแทนอดีตเป็นส่วนข้ามคืนและระหว่างวัน · ใช้เฉพาะวันก่อน days"""
    d = daily.sort_index()
    on, intra = np.log(d["open"] / d["close"].shift(1)), np.log(d["close"] / d["open"])
    rows = {"fo_on_1": on, "fo_id_1": intra}
    for n in (5, 20):
        rows[f"fo_on_mean{n}"] = on.rolling(n, min_periods=n // 2).mean()
        rows[f"fo_id_mean{n}"] = intra.rolling(n, min_periods=n // 2).mean()
    return pd.DataFrame({k: last_before(v, days) for k, v in rows.items()}, index=days)


def market_feature(set50_close, days):
    """F1: ผลตอบแทน SET50 ของวันทำการก่อนหน้า"""
    return last_before(np.log(set50_close.sort_index()).diff(), days)


def rv_by_day(bars):
    """√Σ (ผลตอบแทนแท่ง 60m)² ต่อวัน · ไม่รวมช่วงข้ามคืน → Series ตามวันที่ (ไม่มี tz)"""
    b = bars.sort_values("bar_end")
    day = naive_days(b["date"])
    r = np.log(b["close"].astype(float)).groupby(day.to_numpy()).diff()
    s = np.sqrt((r ** 2).groupby(day.to_numpy()).sum(min_count=1))
    s.index = pd.DatetimeIndex(s.index)
    return s


def rv_features(bars, rows, rules):
    """F3: rows มี date, session · fv_rv60_am ใช้แท่งของวันนี้ที่จบไม่เกินจุดตัดรอบบ่าย"""
    days = naive_days(rows["date"])
    s = rv_by_day(bars)
    out = pd.DataFrame({"fv_rv60_1": last_before(s, days), "fv_rv60_5": last_before(s.rolling(5, min_periods=3).mean(), days), "fv_rv60_20": last_before(s.rolling(20, min_periods=10).mean(), days)}, index=rows.index)
    am = np.full(len(rows), np.nan)
    b = bars.sort_values("bar_end")
    for i, (date, sess) in enumerate(rows[["date", "session"]].itertuples(index=False)):
        if sess != "afternoon":
            continue
        cut = session_times(rules, sess, date)["cutoff"]
        c = np.log(b.loc[(b["date"] == date) & (b["bar_end"] <= cut), "close"].astype(float))
        if len(c) >= 2:
            am[i] = float(np.sqrt((c.diff() ** 2).sum()))
    out["fv_rv60_am"] = am
    return out


def load_daily(symbol):
    d = pd.read_csv(RAW_DIR / "settrade" / f"{symbol}_1d.csv", parse_dates=["timestamp"])
    d.index = pd.DatetimeIndex(d["timestamp"]).tz_localize(None).normalize()
    return d[~d.index.duplicated(keep="last")][["open", "close"]].astype(float)


def add_candidates(d, rules):
    s50 = pd.read_csv(RAW_DIR / "external" / "SET50_1d.csv", index_col=0, parse_dates=True)
    s50.index = pd.DatetimeIndex(s50.index).tz_localize(None).normalize()
    close = s50["Close"].dropna()
    parts = []
    for sym, g in d.groupby("symbol"):
        g = g.copy()
        days = naive_days(g["date"])
        g["fm_set50_1d"] = market_feature(close, days)
        f2 = daily_features(load_daily(sym), days)
        for c in F2:
            g[c] = f2[c].to_numpy()
        f3 = rv_features(load_bars(sym, rules), g, rules)
        for c in F3 + F3_PM:
            g[c] = f3[c].to_numpy()
        parts.append(g)
    return pd.concat(parts).sort_values(KEYS).reset_index(drop=True)


def with_features(var, extra):
    """variant + feature · features แบบ dict แยกรอบ แบบ list รวมทุกรอบ"""
    v = copy.deepcopy(var)
    if isinstance(v["features"], dict):
        v["features"] = {s: list(v["features"][s]) + extra[s] for s in v["features"]}
    else:
        v["features"] = list(v["features"]) + list(dict.fromkeys(extra["morning"] + extra["afternoon"]))
    return v


def run_job(key, d, var, rules):
    return key, run_variant(d, var, rules)[KEYS + ["p_up"]]


def predict_abs_r(g, cols, blocks):
    """|r| ที่ทาย: Ridge แยกหุ้นในรอบเดียว ฝึกจากวันก่อนบล็อก · ก่อนฝึกครบใช้ค่าเฉลี่ยอดีต"""
    out = []
    for sym, h in g.groupby("symbol"):
        h = h.sort_values("date").reset_index(drop=True)
        dates, y, X = h["date"].to_numpy(), np.abs(np.log(h["p_end"].astype(float) / h["p_start"].astype(float))).to_numpy(), h[cols].to_numpy(float)
        pred = pd.Series(y).expanding().mean().shift(1).to_numpy()
        for te in blocks:
            start = np.min(te)
            a, b = (dates < start) & np.isfinite(y), np.isin(dates, te)
            if a.sum() >= 120 and b.any():
                m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0)).fit(X[a], y[a])
                pred[b] = np.clip(m.predict(X[b]), 0, None)
        out.append(h[KEYS].assign(abs_r=pred))
    return pd.concat(out, ignore_index=True)


def importance_job(d, var, session, rules, extra, seed=0):
    """permutation importance: เพิ่ม log loss เมื่อสุ่มสลับ feature บน 200 วันทดสอบท้าย"""
    ev = rules["evaluation"]
    g = d[d["session"] == session].sort_values(KEYS).reset_index(drop=True)
    X = design_matrix(g, select_features(g, var), var["group_by"])
    dates = g["date"].to_numpy()
    ud = np.unique(dates)
    usable = ud[:len(ud) - ev["holdout_days"]]
    start = usable[-200]
    tr = (dates < start) & g["up"].notna().to_numpy()
    te = (dates >= start) & np.isin(dates, usable) & g["direction"].notna().to_numpy() & (g["direction"].to_numpy() != 0)
    m = get_model(var["model"], rules["quantiles"], **var.get("params", {})).fit(X[tr], g["up"].to_numpy()[tr], g["z"].to_numpy()[tr], g[tr])
    y = (g["direction"].to_numpy()[te] == 1).astype(float)
    ll = lambda p: float(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
    base = ll(m.predict(X[te], g[te])["p_up"].to_numpy())
    rng = np.random.default_rng(seed)
    rows = []
    for c in X.columns:
        d_ll = []
        for _ in range(5):
            Xp = X[te].copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            d_ll.append(ll(m.predict(Xp, g[te])["p_up"].to_numpy()) - base)
        rows.append({"session": session, "feature": c, "new": c in extra, "d_ll": float(np.mean(d_ll))})
    t = pd.DataFrame(rows).sort_values("d_ll", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def price_verdict(t, names):
    q, rows = t.set_index(["model", "group"]), []
    for n in names:
        for _, th in SESSIONS:
            a = q.loc[(n, th)]
            k1 = bool(a["d_rmse_pra_hi"] < 0)
            k2 = all(q.loc[(n, f"{s} {th}"), "d_rmse_pra_lo"] <= 0 for s in ("KBANK", "ADVANC"))
            k3 = bool(a["d_mae_pra_lo"] <= 0)
            rows.append({"model": n, "session": th, "rmse": a["rmse"], "mae": a["mae"], "d_rmse": a["d_rmse_pra"], "lo": a["d_rmse_pra_lo"], "hi": a["d_rmse_pra_hi"], "d_mae": a["d_mae_pra"], "mae_lo": a["d_mae_pra_lo"], "mae_hi": a["d_mae_pra_hi"], "c1_rmse": k1, "c2_stocks": k2, "c3_mae": k3, "pass": k1 and k2 and k3})
    return pd.DataFrame(rows)


def direction_verdict(cmp_table, a, b, th):
    q = cmp_table[(cmp_table["a"] == a) & (cmp_table["b"] == b)].set_index("group")
    m = q.loc[th]
    stocks = all(q.loc[f"{s} {th}", "d_ll_lo"] <= 0 for s in ("KBANK", "ADVANC"))
    return bool(m["d_ll_hi"] < 0 and m["d_acc_hi"] >= 0 and stocks)


def main():
    t0 = time.perf_counter()
    s48, s47, s18, s33, s42, s24 = [_module(n) for n in ["48_e23_skip_fx.py", "47_e22_shrink_k.py", "18_score_round4.py", "33_e15_models.py", "42_overfit_audit.py", "51_e24_two_step.py"]]
    rules = load_rules()
    tz, ev, n_boot = rules["timezone"], rules["evaluation"], rules["evaluation"]["n_boot"]
    ds, exps, s31, _, s38, _ = s48.prepare(rules)
    d0, _ = s31.aligned(ds, exps)
    d = add_candidates(d0, rules)
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    print("ค่าว่าง:", d[F1 + F2 + F3].isna().mean().round(4).to_dict(), "· am บ่าย", round(float(d.loc[d["session"] == "afternoon", "fv_rv60_am"].isna().mean()), 4), flush=True)
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    fd = next(x for x in exps["E12"]["variants"] if x["name"] == "fd_lr")
    fd = {**fd, "params": {**fd.get("params", {}), "permutation": False}}
    base = {"eq": v["aprice_lr"], "up": v["pprice_lr"], "flat": v["adir_lr"], "ud": fd}
    jobs = [("ud", "base", fd)]
    for g, extra in GROUPS.items():
        for k, var in base.items():
            jobs.append((k, g, with_features(var, extra)))
    res = Parallel(n_jobs=-1)(delayed(run_job)((k, g), d, var, rules) for k, g, var in jobs)
    pred = dict(res)
    print(f"ตารางฝึกเสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)

    feats = s33.feature_sets(exps)
    mlp = {}
    for g in DIRECTION_GROUPS:
        cols = {s: list(dict.fromkeys(feats[s] + GROUPS[g][s])) for s, _ in SESSIONS}
        p, _, _ = s33.run(d, cols, rules, "up_notup", models=["mlp"])
        mlp[g] = p[p["split"] == "wf"][KEYS + ["p"]]
    print(f"MLP เสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)

    # ทิศทาง
    un = s42.e0_wf(d, "up_notup", rules).rename(columns={"e0": "p_E0-un"})
    aud = s18.load("C1235", "E12_fd_lr", tz)[KEYS + ["p_up"]].rename(columns={"p_up": "p_A-ud"})
    e15 = s42.load_pred(s42.EXP / "E15" / "predictions.csv", tz)
    base_mlp = e15[(e15["model"] == "mlp") & (e15["split"] == "wf")][KEYS + ["p"]].rename(columns={"p": "p_MLP"})
    y = d[KEYS + ["direction"]].merge(un[KEYS + ["p_E0-un"]], on=KEYS).merge(aud, on=KEYS)
    y = y.merge(base_mlp, on=KEYS, how="left")
    for g in DIRECTION_GROUPS:
        y = y.merge(pred[("ud", g)].rename(columns={"p_up": f"p_A-ud+{g}"}), on=KEYS).merge(mlp[g].rename(columns={"p": f"p_MLP+{g}"}), on=KEYS, how="left")
    y = y[y["direction"].notna()].sort_values(KEYS).reset_index(drop=True)
    check = y.merge(pred[("ud", "base")], on=KEYS)
    corr = float(np.corrcoef(check["p_up"], check["p_A-ud"])[0, 1])
    print("fd_lr เทียบ C1235 correlation", round(corr, 6), flush=True)
    assert corr > 0.999, corr
    am, pm = y[y["session"] == "morning"].reset_index(drop=True), y[(y["session"] == "afternoon") & y["p_MLP"].notna() & y["p_MLP+F1"].notna() & y["p_MLP+F2"].notna()].reset_index(drop=True)
    rows, verd = [], []
    for g in DIRECTION_GROUPS:
        for frame, a, b, th in [(am, f"A-ud+{g}", "A-ud", "เช้า"), (pm, f"MLP+{g}", "MLP", "บ่าย")]:
            c = s31.compare(frame, a, b, CONF, n_boot)
            rows.append(c)
            verd.append({"group": g, "session": th, "a": a, "b": b, "pass": direction_verdict(c, a, b, th)})
            rows.append(s31.compare(frame, a, "E0-un", CONF, n_boot))
    dc = pd.concat(rows, ignore_index=True)
    dc.to_csv(RES / "direction.csv", index=False, encoding="utf-8-sig")
    dv = pd.DataFrame(verd)
    dv.to_csv(RES / "direction_verdict.csv", index=False, encoding="utf-8-sig")

    # ราคา
    s47.CONF = CONF
    x = pd.read_csv(OUTPUT_DIR / "experiments" / "E24" / "points.csv")
    x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_convert(tz)
    x = x[KEYS + ["p_ref", "p_end", "p_flat", "p_ud", "r_up", "r_dn", "pt_PR-A", "pt_P(c)", "pt_S0-mean"]].copy()
    assert len(x) == 2188
    pc = x["p_ref"].to_numpy(float)
    names = ["S0-mean"]
    for g in GROUPS:
        z = x[KEYS].merge(pred[("eq", g)].rename(columns={"p_up": "eq"}), on=KEYS, how="left").merge(pred[("up", g)].rename(columns={"p_up": "up"}), on=KEYS, how="left").merge(pred[("flat", g)].rename(columns={"p_up": "flat"}), on=KEYS, how="left").merge(pred[("ud", g)].rename(columns={"p_up": "ud"}), on=KEYS, how="left")
        assert z[["eq", "up", "flat", "ud"]].notna().all().all(), g
        x[f"pt_PR-A+{g}"] = s18.price_rule(pc, z["eq"].to_numpy(float), z["up"].to_numpy(float))[0]
        x[f"pt_S0-mean+{g}"] = pc * np.exp(s24.expected_move(z["flat"].to_numpy(float), z["ud"].to_numpy(float), x["r_up"].to_numpy(float), x["r_dn"].to_numpy(float)))
        names += [f"PR-A+{g}", f"S0-mean+{g}"]
        if g == "F3":
            parts = []
            for s, _ in SESSIONS:
                h = d[(d["session"] == s) & d["p_start"].notna()]
                blocks = [te for _, te in walk_forward_splits(h["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
                parts.append(predict_abs_r(h, GROUPS["F3"][s], blocks))
            ar = x[KEYS].merge(pd.concat(parts, ignore_index=True), on=KEYS, how="left")["abs_r"].to_numpy(float)
            f, u = z["flat"].to_numpy(float), z["ud"].to_numpy(float)
            x["pt_S0-vol"] = pc * np.exp((1 - f) * (2 * u - 1) * ar)
            names.append("S0-vol")
    assert x[[f"pt_{n}" for n in names]].notna().all().all()
    x.to_csv(OUT / "points.csv", index=False)
    t = s47.score(x, ["PR-A", "P(c)"] + names, n_boot, s38.boot_rmse, s31.boot_ci)
    t.to_csv(RES / "price.csv", index=False, encoding="utf-8-sig")
    pv = price_verdict(t, names)
    pv.to_csv(RES / "price_verdict.csv", index=False, encoding="utf-8-sig")
    print(f"คะแนนราคาเสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)

    # เทียบ S0-mean (รายงาน)
    rows = []
    for g in GROUPS:
        for s, th in SESSIONS:
            h = x[x["session"] == s]
            e, b, dates = (h["p_end"] - h[f"pt_S0-mean+{g}"]).to_numpy(float), (h["p_end"] - h["pt_S0-mean"]).to_numpy(float), h["date"].astype(str).to_numpy()
            rows.append(dict(zip(["d_rmse", "lo", "hi"], s38.boot_rmse(e, b, dates, CONF, n_boot)), group=g, session=th))
    s0 = pd.DataFrame(rows)
    s0.to_csv(RES / "price_vs_s0.csv", index=False, encoding="utf-8-sig")

    imp = Parallel(n_jobs=-1)(delayed(importance_job)(d, with_features(fd, GROUPS[g]), s, rules, GROUPS[g][s]) for g in GROUPS for s, _ in SESSIONS)
    ik = [(g, s) for g in GROUPS for s, _ in SESSIONS]
    it = pd.concat([t_.assign(group=k[0]) for k, t_ in zip(ik, imp)], ignore_index=True)
    it.to_csv(RES / "importance.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    print(dc[dc["group"].isin(["เช้า", "บ่าย"])][["a", "b", "group", "n", "acc_a", "acc_b", "pred_up_a", "ll_a", "ll_b", "d_ll", "d_ll_lo", "d_ll_hi", "d_acc", "d_acc_lo", "d_acc_hi"]].round(4).to_string(index=False))
    print(dv.to_string(index=False))
    print(pv.round(4).to_string(index=False))
    print(s0.round(4).to_string(index=False))
    print(it[it["new"]].sort_values(["group", "session", "rank"]).round(5).to_string(index=False))
    print(it.groupby(["group", "session"]).size().to_string())
    print(f"รวม {time.perf_counter() - t0:.0f} วิ", flush=True)


if __name__ == "__main__":
    main()
