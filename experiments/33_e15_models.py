"""E15 (รอบ 7 ข้อ 7) และ engine ของ E17b: nested walk-forward หลายโมเดล — เกณฑ์ใน results/e15_preregistration.md

    python experiments/33_e15_models.py --one     # วัดเวลา: จูน + fit 1 ครั้งต่อโมเดล
    python experiments/33_e15_models.py           # รันเต็ม
→ outputs/experiments/E15/*.csv, results/e15/*.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import itertools
import json
import sys
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import tick_size
from forecast.evaluation.runner import load_or_build_dataset
from forecast.evaluation.splits import holdout_split, walk_forward_splits
from forecast.models.e9 import target_rows
from forecast.models.nested_tune import MODELS, N_TRIALS, RETUNE_DAYS, build, logloss, proba, rank_features, tune

warnings.filterwarnings("ignore")


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


KEYS = ["date", "symbol", "session"]
BLOCKS_PER_TUNE = RETUNE_DAYS // 5


def period_job(d, cols, session, target, name, period, blocks, holdout_dates, seed=0):
    """จูน 1 ครั้ง ณ ต้นช่วง แล้ว fit ใหม่ทุกบล็อกในช่วงนั้น → (แถวผล, ข้อมูลการจูน, out-of-fold)"""
    warnings.filterwarnings("ignore")
    g = d[d["session"] == session].reset_index(drop=True)
    X = g[cols].to_numpy(float)
    m, y_all = target_rows(g, target)
    idx = np.flatnonzero(m)
    y = np.full(len(g), -1)
    y[idx] = y_all
    dates = g["date"].to_numpy()
    start = np.min(blocks[0])
    tr = np.flatnonzero((dates < start) & (y >= 0))
    t0 = time.perf_counter()
    res = tune(name, X[tr], y[tr], dates[tr], g["d_sym_advanc"].to_numpy()[tr], seed)
    tune_sec = time.perf_counter() - t0
    info = {"session": session, "target": target, "model": name, "period": period, "start": str(start)[:10], "tune_sec": tune_sec}
    if res is None:
        return pd.DataFrame(), info, pd.DataFrame()
    info.update({k: res[k] for k in ("params", "k", "inner_ll", "e0_ll", "n_trials")})
    order = rank_features(name, X[tr], y[tr], dates[tr], seed) if name != "enet" else np.arange(X.shape[1])
    use = order if res["k"] is None else order[:res["k"]]
    info["features"] = [cols[j] for j in use]
    oof = g.loc[tr[list(res["oof"].keys())], KEYS].assign(p=list(res["oof"].values()), y=y[tr[list(res["oof"].keys())]], model=name, period=period, session=session)
    out = []
    for te_dates, tag in [(b, "wf") for b in blocks] + ([(holdout_dates, "holdout")] if holdout_dates is not None else []):
        a = np.flatnonzero((dates < np.min(te_dates)) & (y >= 0))
        if len(np.unique(y[a])) < 2:
            continue
        model = build(name, res["params"], seed).fit(X[a][:, use], y[a])
        t = np.flatnonzero(np.isin(dates, te_dates))
        out.append(g.loc[t, KEYS].assign(p=proba(model, X[t][:, use]), model=name, period=period, split=tag))
    return pd.concat(out, ignore_index=True), info, oof


def jobs_for(d, rules, sessions=("morning", "afternoon")):
    ev = rules["evaluation"]
    jobs = []
    for session in sessions:
        dates = d.loc[d["session"] == session, "date"].to_numpy()
        blocks = [te for _, te in walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
        ho = next(te for _, te in holdout_split(dates, ev["holdout_days"]))
        n = int(np.ceil(len(blocks) / BLOCKS_PER_TUNE))
        for p in range(n):
            part = blocks[p * BLOCKS_PER_TUNE:(p + 1) * BLOCKS_PER_TUNE]
            jobs.append((session, p, part, ho if p == n - 1 else None))
    return jobs


def run(d, cols_by_session, rules, target, models=MODELS, one=False, n_jobs=-1):
    jobs = jobs_for(d, rules)
    if one:
        jobs = [j for j in jobs if j[0] == "afternoon"][-1:]
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(period_job)(d, cols_by_session[s], s, target, m, p, part, ho)
                                             for (s, p, part, ho), m in itertools.product(jobs, models))
    preds = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    info = pd.DataFrame([r[1] for r in res])
    oof = pd.concat([r[2] for r in res if len(r[2])], ignore_index=True)
    return preds, info, oof


def ensembles(preds, info, wf_only=True):
    """ENS-2 / all / w / top จากผลทำนายที่มีอยู่ ไม่ฝึกใหม่"""
    w = preds.pivot_table(index=KEYS + ["period", "split"], columns="model", values="p").reset_index()
    ms = [m for m in MODELS if m in w]
    out = {"ENS-2": w[["lr", "mlp"]].mean(axis=1), "ENS-all": w[ms].mean(axis=1)}
    wsoft, wtop = [], []
    im = info.set_index(["session", "period", "model"])
    for _, r in w.iterrows():
        ll = np.array([im.loc[(r["session"], r["period"], m), "inner_ll"] for m in ms], float)
        e0 = np.array([im.loc[(r["session"], r["period"], m), "e0_ll"] for m in ms], float)
        p = r[ms].to_numpy(float)
        s = np.exp(-ll) / np.exp(-ll).sum()
        wsoft.append(float(np.nansum(s * p)))
        wtop.append(float(p[ll < e0].mean()) if (ll < e0).any() else np.nan)
    out["ENS-w"], out["ENS-top"] = wsoft, wtop
    e = w[KEYS + ["period", "split"]].copy()
    long = []
    for name, v in out.items():
        long.append(e.assign(model=name, p=np.asarray(v, float)))
    return pd.concat(long, ignore_index=True)


def inner_scores(oof, info, session):
    """inner-CV log loss ต่อการจูน (out-of-fold ของ best trial) ของแต่ละโมเดลและ ensemble ที่เข้าแข่ง · เฉพาะการจูนที่ทุกโมเดลมีผล"""
    o = oof[oof["session"] == session]
    rows = []
    for period, g in o.groupby("period"):
        w = g.pivot_table(index=KEYS + ["y"], columns="model", values="p").reset_index()
        ms = [m for m in MODELS if m in w]
        w = w.dropna(subset=ms)
        if not len(w):
            continue
        y = w["y"].to_numpy(float)
        im = info[(info["session"] == session) & (info["period"] == period)].set_index("model")
        ll = {m: logloss(y, w[m].to_numpy()).mean() for m in ms}
        s = np.exp(-im.loc[ms, "inner_ll"].to_numpy(float))
        s /= s.sum()
        ll["ENS-all"] = logloss(y, w[ms].mean(axis=1).to_numpy()).mean()
        ll["ENS-w"] = logloss(y, (w[ms].to_numpy() * s).sum(axis=1)).mean()
        top = [m for m in ms if im.loc[m, "inner_ll"] < im.loc[m, "e0_ll"]]
        if top:
            ll["ENS-top"] = logloss(y, w[top].mean(axis=1).to_numpy()).mean()
        else:
            ll["ENS-top"] = float(np.mean(im.loc[ms, "e0_ll"].to_numpy(float)))
        rows.append({"period": period, **ll})
    return pd.DataFrame(rows)


def pbo(perf, s=10, seed=0):
    """Probability of Backtest Overfitting (CSCV): perf = ตารางวัน × แบบ ของ −log loss รายวัน"""
    arr = perf.to_numpy()
    groups = np.array_split(np.arange(len(arr)), s)
    lam = []
    for comb in itertools.combinations(range(s), s // 2):
        a = np.concatenate([groups[i] for i in comb])
        b = np.concatenate([groups[i] for i in range(s) if i not in comb])
        best = np.nanmean(arr[a], axis=0).argmax()
        r = (np.nanmean(arr[b], axis=0) < np.nanmean(arr[b], axis=0)[best]).mean()
        r = np.clip(r, 1 / arr.shape[1], 1 - 1 / arr.shape[1])
        lam.append(np.log(r / (1 - r)))
    return float((np.array(lam) <= 0).mean())


def boot_ci(v, dates, conf, n_boot, seed=0):
    df = pd.DataFrame({"v": np.asarray(v, float), "d": np.asarray(dates)}).dropna()
    s = df.groupby("d")["v"].agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    b = s["sum"].to_numpy()[idx].sum(1) / s["count"].to_numpy()[idx].sum(1)
    al = (1 - conf) / 2
    return s["sum"].sum() / s["count"].sum(), np.quantile(b, al), np.quantile(b, 1 - al)


GROUPS = [("เช้า", None, "morning"), ("บ่าย", None, "afternoon"), ("KBANK เช้า", "KBANK", "morning"), ("KBANK บ่าย", "KBANK", "afternoon"),
          ("ADVANC เช้า", "ADVANC", "morning"), ("ADVANC บ่าย", "ADVANC", "afternoon")]


def evaluate(x, names, refs, conf, n_boot):
    y, flat = (x["direction"] == 1).to_numpy(float), (x["direction"] == 0).to_numpy()
    rows = []
    for name in names:
        p = x[f"p_{name}"].to_numpy(float)
        pred = p > .5
        ok, ll, br = (pred == (y == 1)).astype(float), logloss(y, p), (p - y) ** 2
        for label, sym, sess in GROUPS:
            m = (x["session"] == sess).to_numpy() & (((x["symbol"] == sym).to_numpy()) if sym else True)
            r = {"model": name, "group": label, "n": int(m.sum()), "acc_up_notup": ok[m].mean(), "acc_flat_excl": ok[m & ~flat].mean(),
                 "acc_flat_wrong": (ok[m] * ~flat[m]).mean(), "logloss": ll[m].mean(), "brier": br[m].mean(), "pred_up": pred[m].mean()}
            for ref in refs:
                q = x[f"p_{ref}"].to_numpy(float)
                dates = x["date"].to_numpy()[m]
                a, lo, hi = boot_ci(ll[m] - logloss(y, q)[m], dates, conf, n_boot)
                b, blo, bhi = boot_ci(ok[m] - ((q > .5) == (y == 1)).astype(float)[m], dates, conf, n_boot)
                r.update({f"d_ll_{ref}": a, f"d_ll_{ref}_lo": lo, f"d_ll_{ref}_hi": hi, f"d_acc_{ref}": b, f"d_acc_{ref}_lo": blo, f"d_acc_{ref}_hi": bhi})
            rows.append(r)
    return pd.DataFrame(rows)


def feature_sets(exps):
    fdh = next(v for v in exps["E12"]["variants"] if v["name"] == "fdh_lr")["features"]
    flat = next(v for v in exps["E9FLAT"]["variants"] if v["name"] == "adir_lr")["features"]
    add = ["d_sym_advanc"]
    return {s: list(dict.fromkeys(list(fdh[s]) + list(flat))) + add for s in ("morning", "afternoon")}


def main():
    one = "--one" in sys.argv
    s17, s18, s31, s30 = _module("17_label_variants.py"), _module("18_score_round4.py"), _module("31_e14_finetune.py"), _module("30_e16_up_notup.py")
    rules = load_rules()
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    d, _ = s31.aligned(ds, load_experiments())
    cols = feature_sets(load_experiments())
    print("feature เช้า", len(cols["morning"]), "บ่าย", len(cols["afternoon"]), "· งานจูน", len(jobs_for(d, rules)) * len(MODELS), flush=True)
    if one:
        t = time.perf_counter()
        preds, info, oof = run(d, cols, rules, "up_notup", one=True)
        wall = time.perf_counter() - t
        print(info[["model", "tune_sec", "n_trials", "inner_ll", "e0_ll", "k"]].round(4).to_string(index=False))
        tot = len(jobs_for(d, rules)) * info["tune_sec"].sum() / len(MODELS)
        print(f"wall {wall:.0f} วิ · ผลรวมเวลาจูน {info['tune_sec'].sum():.0f} วิ · ประมาณรวมทุกงาน {tot / 3600:.2f} ชม.·CPU → บน 32 คอร์ ~{tot / 32 / 60:.0f} นาที (ไม่รวม fit รายบล็อก)")
        return
    OUT, RES = OUTPUT_DIR / "experiments" / "E15", ROOT / "results" / "e15"
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    preds, info, oof = run(d, cols, rules, "up_notup")
    print(f"จูนเสร็จ {(time.perf_counter() - t) / 60:.1f} นาที · trial รวม {int(info['n_trials'].sum())}", flush=True)
    preds.to_csv(OUT / "predictions.csv", index=False)
    info.assign(params=info["params"].astype(str), features=info["features"].map(json.dumps)).to_csv(OUT / "tuning.csv", index=False)
    oof.to_csv(OUT / "oof.csv", index=False)
    all_p = pd.concat([preds, ensembles(preds, info)], ignore_index=True)
    all_p.to_csv(OUT / "predictions_all.csv", index=False)
    wf = all_p[all_p["split"] == "wf"]
    assert str(wf["date"].max())[:10] == "2026-10-12"

    a_un = pd.read_csv(OUTPUT_DIR / "experiments" / "E16" / "a_un_predictions.csv")
    a_un["date"] = pd.to_datetime(a_un["date"], utc=True).dt.tz_convert(tz)
    a_ud = s18.load("C1235", "E12_fd_lr", tz)
    e0_un = s30.e0_window(ds, rules)
    names = MODELS + ["ENS-2", "ENS-all", "ENS-w", "ENS-top"]
    x = d[KEYS + ["direction"]].merge(wf.pivot_table(index=KEYS, columns="model", values="p").add_prefix("p_").reset_index(), on=KEYS)
    for n, p in [("A-un", a_un), ("A-ud", a_ud), ("E0-un", e0_un)]:
        x = x.merge(p[KEYS + ["p_up"]].rename(columns={"p_up": f"p_{n}"}), on=KEYS)
    x = x[x["direction"].notna()].reset_index(drop=True)
    x["p_ENS-top"] = x["p_ENS-top"].fillna(x["p_E0-un"])
    print("เทียบบน", len(x), "รอบ", flush=True)
    n_boot, conf = rules["evaluation"]["n_boot"], 1 - 0.05 / 11
    t1 = evaluate(x, names + ["A-un", "E0-un", "A-ud"], ["A-un", "E0-un", "A-ud"], conf, n_boot)
    t1.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")

    rep = []
    for sess in ["morning", "afternoon"]:
        s = inner_scores(oof, info, sess)
        s.to_csv(RES / f"inner_{sess}.csv", index=False, encoding="utf-8-sig")
        cand = [c for c in s.columns if c != "period"]
        mean = s[cand].mean()
        rep.append({"ช่อง": f"{sess}_dir", "ตัวแทน": mean.idxmin(), "inner": mean.min(), **{f"inner_{c}": mean[c] for c in cand}})
    rep = pd.DataFrame(rep)
    rep.to_csv(RES / "representatives.csv", index=False, encoding="utf-8-sig")

    y = (x["direction"] == 1).to_numpy(float)
    perf = pd.DataFrame({n: -logloss(y, x[f"p_{n}"].to_numpy(float)) for n in names}).assign(d=x["date"].to_numpy(), s=x["session"].to_numpy())
    pb = {s: pbo(perf[perf["s"] == s].drop(columns="s").groupby("d").mean()) for s in ["morning", "afternoon"]}
    err = pd.DataFrame({n: x[f"p_{n}"].to_numpy(float) - y for n in MODELS}).corr()
    err.to_csv(RES / "error_corr.csv", encoding="utf-8-sig")
    feat = info.explode("features").groupby(["session", "features"]).size().rename("n").reset_index()
    feat["share"] = feat["n"] / feat["session"].map(info.groupby("session").size())
    feat.sort_values(["session", "share"], ascending=[True, False]).to_csv(RES / "feature_selected.csv", index=False, encoding="utf-8-sig")

    ho = all_p[all_p["split"] == "holdout"].merge(d[KEYS + ["direction"]], on=KEYS)
    ho = ho[ho["direction"].notna()]
    ho["ok"] = (ho["p"] > .5) == (ho["direction"] == 1)
    ho["ll"] = logloss((ho["direction"] == 1).to_numpy(float), ho["p"].to_numpy())
    ho = ho.groupby(["model", "session"]).agg(n=("ok", "size"), acc=("ok", "mean"), ll=("ll", "mean"), pred_up=("p", lambda s: (s > .5).mean())).reset_index()
    ho.to_csv(RES / "holdout.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    show = ["model", "group", "n", "acc_up_notup", "logloss", "pred_up", "d_ll_E0-un", "d_ll_E0-un_lo", "d_ll_E0-un_hi", "d_acc_E0-un", "d_ll_A-ud", "d_ll_A-ud_hi", "d_acc_A-ud"]
    print(t1[t1["group"].isin(["เช้า", "บ่าย"])][show].round(4).to_string(index=False))
    print("\nตัวแทน\n", rep.round(5).T.to_string())
    print("\nPBO", pb)
    print("\nerror corr\n", err.round(2).to_string())
    print("\nfeature ที่ถูกเลือกบ่อย\n", feat.sort_values(["session", "share"], ascending=[True, False]).groupby("session").head(8).round(2).to_string(index=False))
    print("\nholdout (รายงานเฉย ๆ)\n", ho.round(3).to_string(index=False))
    json.dump({"pbo": pb, "trials": int(info["n_trials"].sum())}, open(RES / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
