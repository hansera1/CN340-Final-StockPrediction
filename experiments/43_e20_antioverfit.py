"""E20: แก้ overfit ทุกกลุ่ม — เกณฑ์ใน results/e20_preregistration.md

    python experiments/43_e20_antioverfit.py --part tab ft cnn     # แบบตาราง, κ ตายตัว, CNN ตรึง backbone
    python experiments/43_e20_antioverfit.py --part s2             # ResNet18 S2 ค่าเข้มงวด (ใช้ GPU หลายชั่วโมง)
    python experiments/43_e20_antioverfit.py --part decide         # ตัดสิน 4 ช่อง + สรุป
→ outputs/experiments/E20/*.csv, results/e20/*.csv · ไม่ใช้ holdout · ไม่แตะระบบ live
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.config import OUTPUT_DIR, PROCESSED_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import images_for, load_or_build_dataset
from forecast.evaluation.splits import walk_forward_splits
from forecast.models.e9 import target_rows
from forecast.models.nested_tune import build, proba

warnings.filterwarnings("ignore")
s17, s18, s31, s32, s33, s42 = [importlib.import_module(n) for n in ["17_label_variants", "18_score_round4", "31_e14_finetune", "32_e17a_price", "33_e15_models", "42_overfit_audit"]]
ll, e0_wf, test_gain, verdict, pbo_row, load_pred = s42.ll, s42.e0_wf, s42.test_gain, s42.verdict, s42.pbo_row, s42.load_pred
OUT, RES = OUTPUT_DIR / "experiments" / "E20", ROOT / "results" / "e20"
KEYS = ["date", "symbol", "session"]
SESSIONS = ("morning", "afternoon")
CONF = 0.997

TREE = {"n": 60, "lr": 0.02, "depth": 2, "leaves": 3, "leaf": 100, "sub": 0.6, "col": 0.4, "l2": 20.0}
PARAMS = {"lr": ("lr", {"C": 0.001}), "enet": ("enet", {"C": 0.01, "l1_ratio": 0.5}), "svm": ("svm", {"C": 0.001}),
          "rf": ("rf", {"n": 200, "depth": 2, "leaf": 100, "mf": 0.3}), "mlp": ("mlp", {"hidden": (4,), "alpha": 30.0}),
          "lgbm": ("lgbm", TREE), "xgb": ("xgb", TREE)}
SK = ["lr", "lr_k10", "enet", "svm", "rf", "mlp"]
ALL = SK + ["lgbm", "xgb"]


def fit_predict_one(name, Xa, y, Xs):
    """→ list ของ p ต่อชุดใน Xs · ใช้ข้อมูล Xa เท่านั้น"""
    if name == "lr_k10":
        Z = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(Xa)
        c = np.nan_to_num(np.abs([np.corrcoef(Z[:, j], y)[0, 1] if Z[:, j].std() > 0 else 0.0 for j in range(Z.shape[1])]))
        use = np.argsort(-c)[:10]
        m = build("lr", {"C": 0.003}, 0).fit(Xa[:, use], y)
        return [proba(m, x[:, use]) for x in Xs]
    kind, p = PARAMS[name]
    m = build(kind, p, 0).fit(Xa, y)
    return [proba(m, x) for x in Xs]


def block_job(d, cols, session, target, te, starts, names):
    g = d[d["session"] == session].reset_index(drop=True)
    mask, y_all = target_rows(g, target)
    y = np.full(len(g), -1)
    y[np.flatnonzero(mask)] = y_all
    dates = g["date"].to_numpy()
    start = np.min(te)
    a = (dates < start) & (y >= 0)
    t = np.isin(dates, te)
    X = g[cols].to_numpy(float)
    sym = g["symbol"].to_numpy()
    rate = {s: y[a & (sym == s)].mean() for s in np.unique(sym)}
    e0_t, e0_a = np.array([rate[s] for s in sym[t]]), np.array([rate[s] for s in sym[a]])
    check = str(start)[:10] in starts[session]
    out = g.loc[t, KEYS].copy()
    tr_rows = {}
    pa = {}
    for n in names:
        p_t, p_a = fit_predict_one(n, X[a], y[a], [X[t], X[a]])
        out[n] = p_t
        pa[n] = p_a
    if len(names) > 1:
        out["ENS"] = out[SK].mean(axis=1)
        out["SHR"] = 0.5 * out["ENS"] + 0.5 * e0_t
    out["E0"] = e0_t
    if check:
        ya = y[a].astype(float)
        base = ll(ya, e0_a).mean()
        for n, p in pa.items():
            tr_rows[n] = base - ll(ya, p).mean()
        if len(names) > 1:
            ens = np.mean([pa[n] for n in SK], axis=0)
            tr_rows["ENS"] = base - ll(ya, ens).mean()
            tr_rows["SHR"] = base - ll(ya, 0.5 * ens + 0.5 * e0_a).mean()
    return out, [{"session": session, "start": str(start)[:10], "model": k, "train_gain": v} for k, v in tr_rows.items()]


def run_tab(d, cols, target, names, rules, starts):
    ev = rules["evaluation"]
    jobs = []
    for s in SESSIONS:
        days = d.loc[d["session"] == s, "date"].to_numpy()
        jobs += [(s, te) for _, te in walk_forward_splits(days, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
    res = Parallel(n_jobs=-1)(delayed(block_job)(d, cols[s], s, target, te, starts, names) for s, te in jobs)
    return pd.concat([r[0] for r in res], ignore_index=True).sort_values(KEYS).reset_index(drop=True), pd.DataFrame([x for r in res for x in r[1]])


def overfit_table(wide, tr, e0, target, family, names):
    rows = []
    for s in SESSIONS:
        for n in names:
            r = test_gain(wide.rename(columns={n: "p"})[KEYS + ["p"]], e0, s)
            t = tr[(tr["session"] == s) & (tr["model"] == n)]
            rows.append({"family": family, "model": n, "target": target, "session": s, "n_ckpt": len(t), "train_gain": t["train_gain"].mean(), **r})
    t = pd.DataFrame(rows)
    t["gap"] = t["train_gain"] - t["test_gain"]
    t["verdict"] = t.apply(verdict, axis=1)
    return t


def part_tab(d, rules, exps):
    starts = s42.starts_by_session()
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")["features"]
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    daily = [c for c in fd["morning"] if c.startswith(("fd_", "fl_dow"))]
    eq, up = list(v["aprice_lr"]["features"]), v["pprice_lr"]["features"]
    sets = {"up_notup": s33.feature_sets(exps),
            "flat_price": {s: list(dict.fromkeys(eq + daily)) + ["d_sym_advanc"] for s in SESSIONS},
            "price_updown": {s: list(dict.fromkeys(list(up[s]) + daily)) + ["d_sym_advanc"] for s in SESSIONS}}
    tables = []
    for target, cols in sets.items():
        t0 = time.perf_counter()
        wide, tr = run_tab(d, cols, target, ALL, rules, starts)
        wide.to_csv(OUT / f"tab_{target}.csv", index=False)
        tables.append(overfit_table(wide, tr, e0_wf(d, target, rules), target, f"E20 {target}", ALL + ["ENS", "SHR"]))
        print(target, f"{time.perf_counter() - t0:.0f} วิ", flush=True)
    t0 = time.perf_counter()
    cols = {s: list(fd[s]) + ["d_sym_advanc"] for s in SESSIONS}
    wide, tr = run_tab(d, cols, "updown", ["lr"], rules, starts)
    wide = wide.rename(columns={"lr": "A-ud-strong"})
    wide.to_csv(OUT / "tab_aud_strong.csv", index=False)
    tables.append(overfit_table(wide, tr.assign(model="A-ud-strong"), e0_wf(d, "updown", rules), "updown", "E20 A-ud-strong", ["A-ud-strong"]))
    print("A-ud-strong", f"{time.perf_counter() - t0:.0f} วิ", flush=True)
    t = pd.concat(tables, ignore_index=True)
    t.to_csv(RES / "train_vs_test.csv", index=False, encoding="utf-8-sig")
    print(t[["family", "model", "session", "train_gain", "test_gain", "test_lo", "test_hi", "gap", "verdict"]].round(4).to_string(index=False), flush=True)


# ---------- E14 / E17a: κ ตายตัว ----------
def _kappa30():
    import forecast.models.finetune_lr as f
    f.KAPPAS = (30.0,)
    f.FIXED_KAPPA = 30.0


def ft_dir(d, shared, s, te):
    _kappa30()
    return s31.block(d, shared, s, "D", te)


def ft_price(d, c, s, t, te):
    _kappa30()
    return s32.block(d, c, s, "D", t, te)


def part_ft(ds, d, rules, exps):
    ev = rules["evaluation"]
    _, shared = s31.aligned(ds, exps)
    dp, cols = s32.prepared(ds, exps)
    jobs = []
    for s in SESSIONS:
        days = d.loc[d["session"] == s, "date"].to_numpy()
        jobs += [(s, te) for _, te in walk_forward_splits(days, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
    t0 = time.perf_counter()
    a = pd.concat(Parallel(n_jobs=-1)(delayed(ft_dir)(d, shared, s, te) for s, te in jobs), ignore_index=True)
    a.to_csv(OUT / "ft_dir.csv", index=False)
    print("D κ30", f"{time.perf_counter() - t0:.0f} วิ", a["kappa"].value_counts().to_dict(), flush=True)
    t0 = time.perf_counter()
    b = pd.concat(Parallel(n_jobs=-1)(delayed(ft_price)(dp, cols[t], s, t, te) for s, te in jobs for t in cols), ignore_index=True)
    b.to_csv(OUT / "ft_price.csv", index=False)
    print("PR-D κ30", f"{time.perf_counter() - t0:.0f} วิ", b["kappa"].value_counts().to_dict(), flush=True)


# ---------- E10: CNN ตรึง backbone ----------
def embed(arch, imgs, dev, batch=64):
    import torch

    from forecast.models.finetune import _fresh_net, _prep
    path = PROCESSED_DIR / f"e20_emb_{arch}.npy"
    if path.exists():
        return np.load(path)
    net = _fresh_net(arch, 0).to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), batch):
            x = torch.from_numpy(np.ascontiguousarray(imgs[s:s + batch]))
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(net.features(_prep(x, net.size, dev)).float().cpu().numpy())
    e = np.concatenate(out)
    np.save(path, e)
    return e


def cnn_block(emb, g, y, te, start_check):
    dates = g["date"].to_numpy()
    start = np.min(te)
    a, t = (dates < start) & (y >= 0), np.isin(dates, te)
    m = make_pipeline(StandardScaler(), PCA(n_components=10, random_state=0), LogisticRegression(C=0.01, max_iter=2000)).fit(emb[a], y[a])
    out = g.loc[t, KEYS].assign(p=m.predict_proba(emb[t])[:, 1])
    if str(start)[:10] not in start_check:
        return out, None
    sym = g["symbol"].to_numpy()
    ya = y[a].astype(float)
    r = pd.Series(ya).groupby(sym[a]).transform("mean").to_numpy()
    return out, {"start": str(start)[:10], "train_gain": ll(ya, r).mean() - ll(ya, m.predict_proba(emb[a])[:, 1]).mean()}


def part_cnn(ds, rules):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = rules["evaluation"]
    t0 = time.perf_counter()
    dd = ds.reset_index(drop=True)
    imgs = images_for(dd, rules, "scalogram", tag="_e20wf")
    print("ภาพ", imgs.shape, f"{time.perf_counter() - t0:.0f} วิ", flush=True)
    starts = s42.starts_by_session()
    e0 = e0_wf(dd, "up_notup", rules)
    rows, preds = [], []
    for arch in ["resnet18", "alexnet", "densenet121", "inception_v3", "vgg16_bn"]:
        t0 = time.perf_counter()
        e = embed(arch, imgs, dev)
        for s in SESSIONS:
            idx = np.flatnonzero((dd["session"] == s).to_numpy())
            g = dd.iloc[idx].reset_index(drop=True)
            m, ya = target_rows(g, "up_notup")
            y = np.full(len(g), -1)
            y[np.flatnonzero(m)] = ya
            splits = walk_forward_splits(g["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])
            res = Parallel(n_jobs=-1)(delayed(cnn_block)(e[idx], g, y, te, starts[s]) for _, te in splits)
            p = pd.concat([r[0] for r in res], ignore_index=True).assign(model=f"CNN {arch}")
            preds.append(p)
            tr = [r[1]["train_gain"] for r in res if r[1]]
            rows.append({"family": "E20 CNN ตรึง backbone", "model": arch, "target": "up_notup", "session": s, "n_ckpt": len(tr), "train_gain": np.mean(tr), **test_gain(p, e0, s)})
        print(arch, f"{time.perf_counter() - t0:.0f} วิ", flush=True)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "cnn_frozen.csv", index=False)
    t = pd.DataFrame(rows)
    t["gap"] = t["train_gain"] - t["test_gain"]
    t["verdict"] = t.apply(verdict, axis=1)
    t.to_csv(RES / "cnn_frozen.csv", index=False, encoding="utf-8-sig")
    print(t[["model", "session", "train_gain", "test_gain", "test_lo", "test_hi", "gap", "verdict"]].round(4).to_string(index=False), flush=True)


def part_s2(ds, rules):
    from forecast.models.finetune import CURVES, FineTuneModel
    ev = rules["evaluation"]
    dd = ds.reset_index(drop=True)
    imgs = images_for(dd, rules, "scalogram", tag="_e20wf")
    out = []
    for s in SESSIONS:
        idx = np.flatnonzero((dd["session"] == s).to_numpy())
        g = dd.iloc[idx].reset_index(drop=True)
        dates = g["date"].to_numpy()
        ok = g["direction"].notna().to_numpy()
        for _, te in walk_forward_splits(dates, 20, ev["min_train_days"], ev["holdout_days"]):
            t0 = time.perf_counter()
            a, t = (dates < np.min(te)) & ok, np.isin(dates, te)
            m = FineTuneModel(rules["quantiles"], arch="resnet18", stage="s2", seeds=(0,), epochs=10, patience=2, lr_head=3e-4, lr_block=3e-5, weight_decay=0.2, target="up_notup")
            m.fit(pd.DataFrame(index=g.index[a]), g["up"].to_numpy()[a], g["z"].to_numpy()[a], g[a], imgs[idx][a])
            p = m.predict(pd.DataFrame(index=g.index[t]), g[t], imgs[idx][t])["p_up"].to_numpy()
            out.append(g.loc[t, KEYS].assign(p=p))
            pd.concat(out, ignore_index=True).to_csv(OUT / "s2_strong.csv", index=False)
            pd.DataFrame(CURVES).to_csv(OUT / "s2_strong_curves.csv", index=False)
            print(s, str(np.min(te))[:10], f"{time.perf_counter() - t0:.0f} วิ", flush=True)


# ---------- ตัดสิน ----------
def as_long(wide, names, e0):
    out = {}
    for n in names:
        j = e0.merge(wide[KEYS + [n]], on=KEYS)
        out[n] = j[KEYS].assign(ll=ll(j["_y"].to_numpy(float), j[n]))
    return out


def part_decide(d, ds, rules, tz):
    un = e0_wf(d, "up_notup", rules)
    w = pd.read_csv(OUT / "tab_up_notup.csv")
    w["date"] = pd.to_datetime(w["date"], utc=True).dt.tz_convert(tz)
    aud_s = pd.read_csv(OUT / "tab_aud_strong.csv")
    aud_s["date"] = pd.to_datetime(aud_s["date"], utc=True).dt.tz_convert(tz)
    aud = s18.load("C1235", "E12_fd_lr", tz)
    e15 = load_pred(s42.EXP / "E15" / "predictions.csv", tz)
    mlp = e15[(e15["model"] == "mlp") & (e15["split"] == "wf")]
    x = d[KEYS + ["direction", "p_ref", "p_end", "tick1"]].merge(un[KEYS + ["e0"]].rename(columns={"e0": "p_E0-un"}), on=KEYS)
    x = x.merge(aud[KEYS + ["p_up"]].rename(columns={"p_up": "p_A-ud"}), on=KEYS).merge(mlp[KEYS + ["p"]].rename(columns={"p": "p_MLP"}), on=KEYS, how="left")
    x = x.merge(w[KEYS + ALL + ["ENS", "SHR"]].rename(columns={n: f"p_{n}" for n in ALL + ["ENS", "SHR"]}), on=KEYS)
    x = x.merge(aud_s[KEYS + ["A-ud-strong"]].rename(columns={"A-ud-strong": "p_A-ud-strong"}), on=KEYS)
    ft = pd.read_csv(OUT / "ft_dir.csv")
    ft["date"] = pd.to_datetime(ft["date"], utc=True).dt.tz_convert(tz)
    x = x.merge(ft[KEYS + ["p_up"]].rename(columns={"p_up": "p_D-k30"}), on=KEYS)
    x = x[x["direction"].notna()].sort_values(KEYS).reset_index(drop=True)
    rows = []
    for a in ["SHR", "ENS", "A-ud-strong", "D-k30", *ALL]:
        for b in ["A-ud", "E0-un", "MLP"]:
            h = x if b != "MLP" else x[x["p_MLP"].notna()]
            rows.append(s31.compare(h.reset_index(drop=True), a, b, CONF, 2000))
    c = pd.concat(rows, ignore_index=True)
    c.to_csv(RES / "direction_compare.csv", index=False, encoding="utf-8-sig")
    verdicts = []
    for a in ["SHR"]:
        for sess, th in [("morning", "เช้า"), ("afternoon", "บ่าย")]:
            q = c[(c["a"] == a) & (c["b"] == "A-ud")].set_index("group")
            main = q.loc[th]
            stock_ok = all(q.loc[f"{sym} {th}", "d_ll_lo"] <= 0 and q.loc[f"{sym} {th}", "d_acc_hi"] >= 0 for sym in ("KBANK", "ADVANC"))
            verdicts.append({"ช่อง": f"ทิศทาง{th}", "ตัวแทน": a, "pass_vs_default": bool(main["d_ll_hi"] < 0 and main["d_acc_hi"] >= 0 and stock_ok),
                             "acc": main["acc_a"], "acc_default": main["acc_b"], "pred_up": main["pred_up_a"], "d_ll": main["d_ll"], "lo": main["d_ll_lo"], "hi": main["d_ll_hi"]})
    pd.set_option("display.width", 220)
    print(c[c["group"].isin(["เช้า", "บ่าย"])][["a", "b", "group", "acc_a", "acc_b", "pred_up_a", "d_ll", "d_ll_lo", "d_ll_hi", "d_acc", "d_acc_lo", "d_acc_hi"]].round(4).to_string(index=False), flush=True)

    # ราคา
    s32.CONF = CONF
    feq, fup = [pd.read_csv(OUT / f"tab_{t}.csv") for t in ("flat_price", "price_updown")]
    for f in (feq, fup):
        f["date"] = pd.to_datetime(f["date"], utc=True).dt.tz_convert(tz)
    eqA, upA = [s18.load("C1235", n, tz) for n in ("E9FLAT_aprice_lr", "E9FLAT_pprice_lr")]
    px = d[KEYS + ["p_ref", "p_end"]].merge(eqA[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(upA[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS)
    names = ALL + ["ENS", "SHR"]
    px = px.merge(feq[KEYS + names].rename(columns={n: f"eq_{n}" for n in names}), on=KEYS).merge(fup[KEYS + names].rename(columns={n: f"up_{n}" for n in names}), on=KEYS)
    fp = pd.read_csv(OUT / "ft_price.csv")
    fp["date"] = pd.to_datetime(fp["date"], utc=True).dt.tz_convert(tz)
    for t, pre in [("flat_price", "eq_"), ("price_updown", "up_")]:
        px = px.merge(fp.loc[fp["target"] == t, KEYS + ["p"]].rename(columns={"p": pre + "PR-D-k30"}), on=KEYS)
    px = px[px["p_end"].notna() & px["p_ref"].notna()].sort_values(KEYS).reset_index(drop=True)
    pref, pend = px["p_ref"].to_numpy(float), px["p_end"].to_numpy(float)
    ea = pend - s18.price_rule(pref, px["eq_A"].to_numpy(), px["up_A"].to_numpy())[0]
    groups = [("เช้า", None, "morning"), ("บ่าย", None, "afternoon"), ("KBANK เช้า", "KBANK", "morning"), ("KBANK บ่าย", "KBANK", "afternoon"), ("ADVANC เช้า", "ADVANC", "morning"), ("ADVANC บ่าย", "ADVANC", "afternoon")]
    prows = []
    for n in names + ["PR-D-k30"]:
        e, shift = None, None
        pt, shift = s18.price_rule(pref, px[f"eq_{n}"].to_numpy(), px[f"up_{n}"].to_numpy())
        e = pend - pt
        for label, sym, sess in groups:
            m = (px["session"] == sess).to_numpy()
            if sym:
                m &= (px["symbol"] == sym).to_numpy()
            dates = px["date"].astype(str).to_numpy()[m]
            mae = s33.boot_ci(np.abs(e[m]) - np.abs(ea[m]), dates, CONF, 2000)
            rm = s32.boot_rmse(e[m], ea[m], dates, 2000)
            prows.append({"model": n, "group": label, "n": int(m.sum()), "mae": np.abs(e[m]).mean(), "mae_PR-A": np.abs(ea[m]).mean(), "rmse": np.sqrt((e[m] ** 2).mean()), "rmse_PR-A": np.sqrt((ea[m] ** 2).mean()),
                          "d_mae": mae[0], "d_mae_lo": mae[1], "d_mae_hi": mae[2], "d_rmse": rm[0], "d_rmse_lo": rm[1], "d_rmse_hi": rm[2], "shift_rate": (shift[m] != 0).mean()})
    p = pd.DataFrame(prows)
    p.to_csv(RES / "price_compare.csv", index=False, encoding="utf-8-sig")
    print(p[p["group"].isin(["เช้า", "บ่าย"])][["model", "group", "mae", "mae_PR-A", "rmse", "rmse_PR-A", "d_mae", "d_mae_lo", "d_mae_hi", "d_rmse", "d_rmse_lo", "d_rmse_hi", "shift_rate"]].round(3).to_string(index=False), flush=True)
    for th in ("เช้า", "บ่าย"):
        q = p[p["model"] == "SHR"].set_index("group")
        r = q.loc[th]
        win = (r["d_mae_hi"] < 0 and r["d_rmse_lo"] <= 0) or (r["d_rmse_hi"] < 0 and r["d_mae_lo"] <= 0)
        stock_ok = all(q.loc[f"{sym} {th}", "d_mae_lo"] <= 0 and q.loc[f"{sym} {th}", "d_rmse_lo"] <= 0 for sym in ("KBANK", "ADVANC"))
        verdicts.append({"ช่อง": f"ราคา{th}", "ตัวแทน": "SHR", "pass_vs_default": bool(win and stock_ok), "acc": np.nan, "acc_default": np.nan, "pred_up": np.nan, "d_ll": r["d_mae"], "lo": r["d_mae_lo"], "hi": r["d_mae_hi"]})
    v = pd.DataFrame(verdicts)
    v.to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")
    print(v.round(4).to_string(index=False), flush=True)

    # PBO กลุ่มแก้แล้ว
    fam = as_long(w, ALL + ["ENS", "SHR", "E0"], un)
    d_fam = dict(fam)
    d_fam["A-ud-strong"] = as_long(aud_s, ["A-ud-strong"], un)["A-ud-strong"]
    d_fam["D-k30"] = as_long(ft.rename(columns={"p_up": "D-k30"}), ["D-k30"], un)["D-k30"]
    rows = pbo_row("E20 ทิศทาง", d_fam)
    cf = OUT / "cnn_frozen.csv"
    if cf.exists():
        c2 = load_pred(cf, tz)
        cnn = {}
        for n, h in c2.groupby("model"):
            j = un.merge(h[KEYS + ["p"]], on=KEYS)
            cnn[n] = j[KEYS].assign(ll=ll(j["_y"].to_numpy(float), j["p"]))
        cnn["E0"] = fam["E0"]
        rows += pbo_row("E20 CNN ตรึง backbone", cnn)
    eq0, up0 = e0_wf(d, "flat_price", rules), e0_wf(d, "price_updown", rules)
    pfam = {}
    for n in ALL + ["ENS", "SHR", "E0"]:
        a, b = as_long(feq, [n], eq0)[n], as_long(fup, [n], up0)[n]
        pfam[n] = pd.concat([a, b], ignore_index=True)
    rows += pbo_row("E20 ราคา", pfam)
    b = pd.DataFrame(rows)
    b.to_csv(RES / "pbo.csv", index=False, encoding="utf-8-sig")
    print(b.round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    exps = load_experiments()
    d, _ = s31.aligned(ds, exps)
    if "tab" in args.part:
        part_tab(d, rules, exps)
    if "ft" in args.part:
        part_ft(ds, d, rules, exps)
    if "cnn" in args.part:
        part_cnn(ds, rules)
    if "decide" in args.part:
        part_decide(d, ds, rules, tz)
    if "s2" in args.part:
        part_s2(ds, rules)


if __name__ == "__main__":
    main()
