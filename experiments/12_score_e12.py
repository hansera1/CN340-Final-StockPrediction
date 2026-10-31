"""E12 (NEXT_STEPS รอบที่ 3 ข้อ 2.3): ตาราง (F0, F0+F-daily, F0+F-daily+F-60m) × (LR, RF, XGB, MLP)
เทียบ E0-ud และ E9-LRx บนแถวทดสอบเดียวกัน + permutation importance + pilot timeframe

    python experiments/12_score_e12.py [--skip-pilot]
    → results/e12/*.csv

เกณฑ์แทน E9-LRx (ข้อ 3): CI ของผลต่างความแม่นรวม (ไม่นับ flat) เทียบ E9-LRx ไม่คร่อม 0 (ดีกว่า)
และไม่มีชุดหุ้น/รอบไหนที่แย่กว่า E9-LRx อย่างมีนัยสำคัญ — ถ้าผ่านหลายตัว เลือกตัวที่ง่ายที่สุด
"""
import argparse
import copy

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import row_losses
from forecast.evaluation.runner import load_or_build_dataset, run_variant
from forecast.features.cn340 import asof_bar_features
from forecast.data.io import load_bars

import importlib.util
_spec = importlib.util.spec_from_file_location("s09", ROOT / "experiments" / "09_score_round2.py")
s09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s09)

OUT = ROOT / "results" / "e12"
SETS = {"f0": "F0", "fd": "F0 + F-daily", "fdh": "F0 + F-daily + F-60m"}
MODELS = {"lr": "LR", "rf": "RF", "xgb": "XGB", "mlp": "MLP"}
SIMPLICITY = ["lr", "rf", "xgb", "mlp"]          # ง่าย → ซับซ้อน (ใช้ตัดสินเมื่อผ่านหลายตัว)


def common_rows(preds):
    keys = None
    for p in preds.values():
        k = set(map(tuple, p[s09.KEYS].astype(str).to_numpy()))
        keys = k if keys is None else keys & k
    return {n: p[p[s09.KEYS].astype(str).apply(tuple, axis=1).isin(keys)].reset_index(drop=True)
            for n, p in preds.items()}


def importance(p, q):
    """Δ log loss ขึ้น/ลง และ Δ ความแม่น (ไม่นับ flat) เมื่อสุ่มค่า feature ใหม่จากชุดฝึก (นอกตัวอย่าง) — บวก = feature มีประโยชน์"""
    base = row_losses(p, q)
    rows = []
    for c in [c for c in p if c.startswith("perm_")]:
        pp = p.assign(p_up=p[c].fillna(p["p_up"]))
        l = row_losses(pp, q)
        for sess in ["morning", "afternoon"]:
            idx = base.index[(p.loc[base.index, "session"] == sess) & p.loc[base.index, c].notna()]
            if not len(idx):
                continue
            rows.append({"feature": c[5:], "session": sess,
                         "d_logloss_ud": l.loc[idx, "logloss_ud"].mean() - base.loc[idx, "logloss_ud"].mean(),
                         "d_acc_nf": l.loc[idx, "err_nf"].mean() - base.loc[idx, "err_nf"].mean()})
    return pd.DataFrame(rows)


def pilot(rules, ds):
    """F-60m ซ้ำบนแท่ง 30m / 5m (yfinance 60 วัน) — รอบบ่ายเท่านั้น, LR สเปก E9-LR,
    ทั้งสามความละเอียดใช้วันเดียวกัน (เฉพาะวันที่มี 5m) ฝึกขั้นต่ำ 30 วัน ทดสอบทีละ 5 วัน, ไม่แตะ holdout"""
    days5 = None
    for sym in rules["symbols"]:
        b = load_bars(sym, rules, None, "5m")
        d = set(b["date"].unique())
        days5 = d if days5 is None else days5 & d
    d = ds[(ds["session"] == "afternoon") & ds["date"].isin(days5)].copy()
    for iv, pre in [("30m", "fh30"), ("5m", "fh5")]:
        bars = {s: load_bars(s, rules, None, iv) for s in rules["symbols"]}
        d = d.join(asof_bar_features(d, bars, pre))
    exp = load_experiments()["E12"]
    base = next(v for v in exp["variants"] if v["name"] == "fdh_lr")
    fd_aft = [c for c in base["features"]["afternoon"] if not c.startswith("fh_")]
    r = copy.deepcopy(rules)
    r["evaluation"]["min_train_days"] = 30
    preds = {}
    for tf, pre in [("60m", "fh"), ("30m", "fh30"), ("5m", "fh5")]:
        cols = fd_aft + [f"{pre}_{k}" for k in ["rsi14", "macd_hist", "bb_pos", "c_sma20", "atr14"]]
        v = dict(base, name=f"pilot_{tf}", sessions=["afternoon"], features={"afternoon": cols},
                 params={"permutation": False})
        preds[tf] = run_variant(d.reset_index(drop=True), v, r)
    v0 = {"name": "pilot_e0ud", "model": "baseline", "params": {"target": "updown"}, "sessions": ["afternoon"]}
    preds["E0-ud"] = run_variant(d.reset_index(drop=True), v0, r)
    preds = common_rows(preds)
    q = rules["quantiles"]
    rows = []
    for name, p in preds.items():
        s = s09.scores(p, q)
        rows.append({"timeframe": name, "n_days": p["date"].nunique(), "n_nonflat": s["n_nonflat"],
                     "acc_nf": s["acc_nf"], "logloss_ud": s["logloss_ud"],
                     **(s09.boot(p, preds["E0-ud"], q, r["evaluation"], None) if name != "E0-ud" else {})})
    out = pd.DataFrame(rows)
    out.insert(0, "note", "PILOT: รอบบ่าย, 60 วัน, ฝึกขั้นต่ำ 30 วัน — ไม่ใช้เลือกโมเดล")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pilot", action="store_true")
    args = ap.parse_args()
    rules = load_rules()
    tz, q, ev = rules["timezone"], rules["quantiles"], rules["evaluation"]
    OUT.mkdir(parents=True, exist_ok=True)

    preds = {"E0-ud": s09.load("E9", "baseline_ud", tz), "E9-LRx": s09.load("E9", "lrx", tz)}
    for fs in SETS:
        for m in MODELS:
            p = s09.load("E12", f"{fs}_{m}", tz)
            if p is not None:
                preds[f"{fs}_{m}"] = p
    preds = common_rows(preds)

    rows, conv = [], []
    for name, p in preds.items():
        for g in s09.GROUPS:
            sub = s09.subset(p, g)
            row = {"model": name, "features": SETS.get(name.split("_")[0], ""), "algo": MODELS.get(name.split("_")[-1], ""),
                   "group": s09.gname(g), **s09.scores(sub, q)}
            if name != "E0-ud":
                row.update({f"vsE0ud_{k}": v for k, v in s09.boot(p, preds["E0-ud"], q, ev, g).items()})
            if name not in ("E0-ud", "E9-LRx"):
                row.update({f"vsLRx_{k}": v for k, v in s09.boot(p, preds["E9-LRx"], q, ev, g).items()})
            rows.append(row)
        if "mlp_converged" in p:
            conv.append({"model": name, "share_converged": p["mlp_converged"].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "e12_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conv).to_csv(OUT / "mlp_convergence.csv", index=False, encoding="utf-8-sig")

    # เกณฑ์แทน E9-LRx
    cand = []
    for name in [n for n in preds if n not in ("E0-ud", "E9-LRx")]:
        a = t[(t.model == name) & (t.group == "ALL")].iloc[0]
        per = t[(t.model == name) & (t.group != "ALL")]
        better = a["vsLRx_d_acc_nf_lo"] > 0
        worse_any = bool((per["vsLRx_d_acc_nf_hi"] < 0).any())
        cand.append({"model": name, "d_acc_nf_vs_LRx": a["vsLRx_d_acc_nf"], "lo": a["vsLRx_d_acc_nf_lo"],
                     "hi": a["vsLRx_d_acc_nf_hi"], "ci_excludes_0_better": better,
                     "any_group_sig_worse": worse_any, "passes": better and not worse_any})
    cand = pd.DataFrame(cand)
    cand.to_csv(OUT / "replace_lrx_check.csv", index=False, encoding="utf-8-sig")

    # importance ของตัวที่ดีที่สุด (ความแม่นรวมไม่นับ flat) ในแต่ละชุด feature
    imp = []
    for fs in SETS:
        names = [n for n in preds if n.startswith(fs + "_")]
        if not names:
            continue
        best = max(names, key=lambda n: t[(t.model == n) & (t.group == "ALL")]["acc_nf"].iat[0])
        i = importance(preds[best], q)
        i.insert(0, "model", best)
        imp.append(i)
    if imp:
        pd.concat(imp).to_csv(OUT / "importance.csv", index=False, encoding="utf-8-sig")

    show = t[t.group == "ALL"][["model", "acc_nf", "logloss_ud", "vsE0ud_d_acc_nf", "vsE0ud_d_acc_nf_lo",
                                "vsE0ud_d_acc_nf_hi", "vsLRx_d_acc_nf", "vsLRx_d_acc_nf_lo", "vsLRx_d_acc_nf_hi"]]
    print(show.round(4).to_string(index=False))
    print(cand.round(4).to_string(index=False))
    print(pd.DataFrame(conv).to_string(index=False))
    if not args.skip_pilot:
        ds = load_or_build_dataset(rules)
        pl = pilot(rules, ds)
        pl.to_csv(OUT / "pilot_timeframe.csv", index=False, encoding="utf-8-sig")
        print(pl.round(4).to_string(index=False))
    print("→", OUT)


if __name__ == "__main__":
    main()
