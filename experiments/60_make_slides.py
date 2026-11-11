"""ข้อมูลทุกตัวเลขของสไลด์ฉบับสุดท้าย — อ่านไฟล์ผลลัพธ์โดยตรง (docs/prompt_slides_final.md)

    python experiments/60_make_slides.py
→ docs/slides/.build/slides_data.json (มีราคาจริงในตารางส่งจริงและกราฟ จึงไม่อยู่ใน git)
  แล้ว node docs/slides/build_deck.js สร้าง docs/slides/CN340_final_presentation.pptx
"""
import copy
import importlib.util
import json
import re
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import tick_size
from forecast.evaluation.runner import load_or_build_dataset, run_variant

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
RES = ROOT / "results"
OUT = ROOT / "docs" / "slides" / ".build"
CONF = 0.997
LIVE_FROM = "2026-11-09"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def r(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), n)


def tz_dates(df, tz):
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
    return df


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def classification(y_up, pred_up, p):
    """confusion matrix + ตัวชี้วัด · Precision ไม่นิยามเมื่อไม่มีการทาย "ขึ้น" """
    y_up, pred_up, p = np.asarray(y_up, bool), np.asarray(pred_up, bool), np.asarray(p, float)
    tp, fp = int((pred_up & y_up).sum()), int((pred_up & ~y_up).sum())
    fn, tn = int((~pred_up & y_up).sum()), int((~pred_up & ~y_up).sum())
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and (prec + rec) > 0 else None
    auc = roc_auc_score(y_up, p) if len(np.unique(y_up)) == 2 and np.ptp(p) > 0 else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / n if n else None, "precision": prec,
            "recall": rec, "specificity": spec, "f1": f1, "balanced_accuracy": (rec + spec) / 2 if rec is not None and spec is not None else None,
            "auc": auc, "logloss": float(ll(y_up.astype(float), p).mean()), "pred_up_rate": float(pred_up.mean()), "up_rate": float(y_up.mean())}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)[:10]
    return o


def main():
    s17, s18, s30, s31 = [_module(n) for n in ["17_label_variants.py", "18_score_round4.py", "30_e16_up_notup.py", "31_e14_finetune.py"]]
    rules = load_rules()
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    exps = load_experiments()
    D = {"sources": {}}

    # ---------- ข้อมูล ----------
    lab = ds[ds["direction"].notna()].copy()
    lab["tick"] = tick_size(lab["p_start"].to_numpy(float))
    lab["move_ticks"] = (lab["p_end"] - lab["p_start"]).abs() / lab["tick"]
    dist = []
    for (sym, sess), g in lab.groupby(["symbol", "session"]):
        dist.append({"symbol": sym, "session": sess, "n": len(g), "up": (g["direction"] == 1).mean(), "flat": (g["direction"] == 0).mean(),
                     "down": (g["direction"] == -1).mean(), "move_ticks": g["move_ticks"].mean()})
    days = np.sort(ds["date"].unique())
    D["data"] = {"first_day": str(days[0])[:10], "last_day": str(days[-1])[:10], "n_days": len(days), "dist": dist,
                 "flat_range": [min(x["flat"] for x in dist), max(x["flat"] for x in dist)],
                 "kbank_flat": pd.read_csv(RES / "e14" / "kbank_flat_rate.csv").to_dict("records")}
    D["sources"]["data"] = "dataset _wf (data/processed) · results/e14/kbank_flat_rate.csv"

    # ---------- ทิศทาง: walk-forward ----------
    aud = s18.load("C1235", "E12_fd_lr", tz)
    e15 = tz_dates(pd.read_csv(OUTPUT_DIR / "experiments" / "E15" / "predictions.csv"), tz)
    mlp = e15[e15["model"] == "mlp"]
    e0 = s30.e0_window(ds, rules)
    x = ds[KEYS + ["direction"]].merge(e0[KEYS + ["p_up"]].rename(columns={"p_up": "p_E0-un"}), on=KEYS)
    x = x.merge(aud[KEYS + ["p_up"]].rename(columns={"p_up": "p_A-ud"}), on=KEYS)
    x = x.merge(mlp[mlp["split"] == "wf"][KEYS + ["p"]].rename(columns={"p": "p_E15 MLP"}), on=KEYS, how="left")
    x = x[x["direction"].notna()].reset_index(drop=True)
    y = (x["direction"] == 1).to_numpy()
    cls = {}
    for sess, model, rule in [("morning", "A-ud", ">="), ("afternoon", "E15 MLP", ">"), ("morning", "E0-un", ">"), ("afternoon", "E0-un", ">"), ("afternoon", "A-ud", ">=")]:
        m = (x["session"] == sess).to_numpy() & x[f"p_{model}"].notna().to_numpy()
        p = x[f"p_{model}"].to_numpy(float)[m]
        pred = p >= 0.5 if rule == ">=" else p > 0.5
        cls[f"{sess}|{model}"] = classification(y[m], pred, p)
        for sym in ("KBANK", "ADVANC"):
            ms = m & (x["symbol"] == sym).to_numpy()
            ps = x[f"p_{model}"].to_numpy(float)[ms]
            cls[f"{sess}|{model}|{sym}"] = classification(y[ms], ps >= 0.5 if rule == ">=" else ps > 0.5, ps)
    D["cls_wf"] = cls
    sig = []
    for sess, a, b in [("morning", "A-ud", "E0-un"), ("afternoon", "E15 MLP", "E0-un"), ("afternoon", "E15 MLP", "A-ud")]:
        h = x[(x["session"] == sess) & x[f"p_{a}"].notna()].reset_index(drop=True)
        c = s31.compare(h, a, b, CONF, rules["evaluation"]["n_boot"])
        sig += c.assign(session=sess).to_dict("records")
    D["sig_wf"] = sig
    D["e16"] = pd.read_csv(RES / "e16" / "metrics.csv").to_dict("records")
    D["sources"]["cls"] = "outputs/experiments/C1235/E12_fd_lr_predictions.csv · outputs/experiments/E15/predictions.csv · e0_window (30_e16_up_notup.py)"

    # holdout
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")
    fd = {**copy.deepcopy(fd), "params": {**fd.get("params", {}), "permutation": False}}
    ho = run_variant(ds, fd, rules, holdout_eval=True)[KEYS + ["direction", "p_up"]]
    hm = ho[(ho["session"] == "morning") & ho["direction"].notna()]
    ha = mlp[mlp["split"] == "holdout"].merge(ds[KEYS + ["direction"]], on=KEYS)
    ha = ha[(ha["session"] == "afternoon") & ha["direction"].notna()]
    D["cls_holdout"] = {"morning|A-ud": classification(hm["direction"] == 1, hm["p_up"] >= 0.5, hm["p_up"]),
                        "afternoon|E15 MLP": classification(ha["direction"] == 1, ha["p"] > 0.5, ha["p"])}

    # ---------- ส่งจริง ----------
    live = tz_dates(pd.read_csv(OUTPUT_DIR / "live_vm" / "live_scored.csv"), tz)
    live = live[live["date"] >= pd.Timestamp(LIVE_FROM, tz=tz)].sort_values(["date", "session", "symbol"], ascending=[True, False, True])
    rows = []
    for _, q in live.iterrows():
        done = pd.notna(q["direction"])
        actual = None if not done else ("ขึ้น" if q["direction"] == 1 else "นิ่ง" if q["direction"] == 0 else "ลง")
        pred = "ขึ้น" if q["answer"] == 1 else "ไม่ขึ้น"
        rows.append({"date": str(q["date"])[:10], "session": q["session"], "symbol": q["symbol"], "pred": pred, "p_up": r(q["p_up"], 3), "actual": actual,
                     "correct": None if not done else bool((q["answer"] == 1) == (q["direction"] == 1)), "point": r(q["point_price"], 2), "p_end": r(q["p_end"], 2),
                     "err": None if not done else r(q["point_price"] - q["p_end"], 2)})
    lv = pd.DataFrame(rows)
    summ = {}
    for sess in ("morning", "afternoon", "all"):
        h = lv[lv["actual"].notna() & ((lv["session"] == sess) if sess != "all" else True)]
        e = h["err"].astype(float)
        summ[sess] = {"n": len(h), "correct": int(h["correct"].sum()), "acc": h["correct"].mean() if len(h) else None,
                      "mae": e.abs().mean() if len(h) else None, "rmse": float(np.sqrt((e ** 2).mean())) if len(h) else None}
    exp_acc = {"morning": cls["morning|A-ud"]["accuracy"], "afternoon": cls["afternoon|E15 MLP"]["accuracy"]}
    expect = {}
    for sess, p in exp_acc.items():
        k = 10
        expect[sess] = {"backtest_acc": p, "n_full_week": k, "mean": p * k, "range95": [max(0, p * k - 1.96 * np.sqrt(k * p * (1 - p))), min(k, p * k + 1.96 * np.sqrt(k * p * (1 - p)))]}
    D["live"] = {"rows": rows, "summary": summ, "expected": expect, "last_day": lv["date"].max(), "days": int(lv["date"].nunique())}
    D["sources"]["live"] = "VM outputs/logs/predictions.jsonl → scripts/06_score_live.py → outputs/live_vm/live_scored.csv"

    # ---------- ราคา ----------
    m22 = pd.read_csv(RES / "e22" / "metrics.csv")
    price = m22[m22["model"].isin(["PR-A", "P(c)", "k=1"])].copy()
    price["model"] = price["model"].replace({"k=1": "PR-mean"})
    D["price"] = price.to_dict("records")
    D["price_e17"] = {"morning": {"model": "E17b ENS-top", "mae": 2.010, "rmse": 3.067, "d_rmse": -0.016, "lo": -0.044, "hi": 0.014},
                      "afternoon": {"model": "E17b MLP", "mae": 1.271, "rmse": 1.817, "d_rmse": -0.013, "lo": -0.032, "hi": 0.002},
                      "source": "results/decision_4slots.md ช่อง 3–4 (CI 99.7% เทียบ PR-A)"}
    cov = pd.read_csv(RES / "e17a" / "coverage.csv")
    D["coverage"] = cov[(cov["model"] == "PR-A") & (cov["quantile"] == "ทุกแถว")].to_dict("records")
    D["oracle"] = pd.read_csv(RES / "e27" / "oracle.csv").to_dict("records")
    D["xd"] = pd.read_csv(RES / "e27" / "xd_effect.csv").to_dict("records")
    D["sources"]["price"] = "results/e22/metrics.csv (PR-A, P(c), PR-mean = k=1) · results/e17a/coverage.csv · results/e27/oracle.csv, xd_effect.csv"
    # กราฟตัวอย่าง: 3 สัปดาห์สุดท้ายของ walk-forward รอบเช้า KBANK
    eq = s18.load("C1235", "E9FLAT_aprice_lr", tz)
    pp = s18.load("C1235", "E9FLAT_pprice_lr", tz)
    zq = s18.load("C1235", "E9_baseline_ud", tz)
    g = eq[KEYS + ["p_ref", "p_end", "p_up"]].rename(columns={"p_up": "eq"}).merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "up"}), on=KEYS)
    g = g.merge(zq[KEYS + ["q10", "q25", "q50", "q75", "q90"]], on=KEYS)
    g = g[g["p_end"].notna()].copy()
    g["pt"] = s18.price_rule(g["p_ref"].to_numpy(float), g["eq"].to_numpy(), g["up"].to_numpy())[0]
    g["err"] = g["pt"] - g["p_end"]
    ex = g[(g["symbol"] == "KBANK") & (g["session"] == "morning")].sort_values("date").tail(15)
    D["price_example"] = [{"date": str(d)[5:10], "actual": a, "pred": p, "lo80": float(round(p * np.exp(q10 - q50), 2)), "hi80": float(round(p * np.exp(q90 - q50), 2))}
                          for d, a, p, q10, q50, q90 in zip(ex["date"], ex["p_end"], ex["pt"], ex["q10"], ex["q50"], ex["q90"])]
    bins = np.arange(-10.5, 11.5, 1.0)
    hist = {}
    for sess in ("morning", "afternoon"):
        e = g.loc[g["session"] == sess, "err"].clip(-10, 10)
        hist[sess] = np.histogram(e, bins=bins)[0].tolist()
    D["err_hist"] = {"bins": [int(b + 0.5) for b in bins[:-1]], **hist}

    # ---------- permutation importance ของ A-ud ----------
    am = aud[(aud["session"] == "morning") & aud["direction"].notna() & (aud["direction"] != 0)]
    yy = (am["direction"] == 1).to_numpy(float)
    base = ll(yy, am["p_up"]).mean()
    imp = sorted(((c[5:], float(ll(yy, am[c]).mean() - base)) for c in am.columns if c.startswith("perm_")), key=lambda t: -t[1])
    D["importance"] = [{"feature": f, "d_logloss": v} for f, v in imp[:12]]
    D["sources"]["importance"] = "คอลัมน์ perm_* ใน outputs/experiments/C1235/E12_fd_lr_predictions.csv (P(ขึ้น) เมื่อสลับค่า feature) · log loss ขึ้น/ลง รอบเช้า"

    # ---------- E15 MLP ที่ล็อก ----------
    import pickle
    b = pickle.load(open(OUTPUT_DIR / "models" / "R4ATC_UN" / "E15" / "mlp" / "KBANK_afternoon.pkl", "rb"))
    mm = b["model"].named_steps["mlpclassifier"]
    D["mlp"] = {"version": b["model_version"], "trained_until": b["trained_until"], "hidden": list(mm.hidden_layer_sizes), "alpha": float(mm.alpha), "features": b["feature_cols"]}
    a = pickle.load(open(OUTPUT_DIR / "models" / "R4ATC" / "E12" / "fd_lr" / "KBANK_morning.pkl", "rb"))
    D["aud"] = {"version": a["model_version"], "trained_until": a["trained_until"], "C": float(a["model"].C_), "n_bags": len(a["model"].models_), "n_features": len(a["feature_cols"])}

    # ---------- การเงิน ----------
    fm = pd.read_csv(RES / "financial" / "metrics.csv")
    D["fin"] = fm.to_dict("records")
    D["fin_diff"] = pd.read_csv(RES / "financial" / "sharpe_diff.csv").to_dict("records")
    D["fin_counts"] = pd.read_csv(RES / "financial" / "signal_counts.csv").to_dict("records")
    for fee in ("0", "on"):
        e = pd.read_csv(RES / "financial" / f"equity_wf_fee{fee}.csv")
        step = max(1, len(e) // 60)
        D[f"equity_fee{fee}"] = e.iloc[::step].to_dict("records")

    # ---------- การทดลองอื่น ----------
    cnn = pd.read_csv(RES / "e20" / "cnnb_finetune.csv")
    D["cnn"] = cnn[["model", "session", "test_gain", "test_lo", "test_hi", "best_epoch_median", "overfit_early"]].to_dict("records")
    D["e28_verdict"] = pd.read_csv(RES / "e28" / "verdict.csv").to_dict("records")
    D["ledger_total"] = int(pd.to_numeric(pd.read_csv(RES / "e28" / "comparison_ledger.csv").iloc[:, 2], errors="coerce").sum())
    lock = (RES / "lock_decision.md").read_text(encoding="utf-8")
    mt = re.search(r"ความแม่นไม่นับ flat ([\d.]+)% เทียบ E0-ud ([\d.]+)% \(([+−-][\d.]+) \[([+−-][\d.]+), ([+−-][\d.]+)\]\)", lock)
    D["aud_updown"] = {"acc": mt.group(1), "e0ud": mt.group(2), "diff": mt.group(3), "lo": mt.group(4), "hi": mt.group(5), "source": "results/lock_decision.md บรรทัด 27 (CI 95%)"} if mt else None
    md = pd.read_csv(RES / "e16_morning_daily" / "summary.csv")
    D["recent32"] = md[md["period"].str.startswith("รวม")].to_dict("records")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slides_data.json").write_text(json.dumps(clean(D), ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", OUT / "slides_data.json")
    for k in ("morning|A-ud", "afternoon|E15 MLP", "morning|E0-un", "afternoon|E0-un"):
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in cls[k].items()})
    print("live", summ, "· days", D["live"]["days"])
    print("aud_updown", D["aud_updown"], "· ledger", D["ledger_total"])
    print("importance", D["importance"][:5])
    print("mlp", D["mlp"]["hidden"], round(D["mlp"]["alpha"], 3), len(D["mlp"]["features"]), "· aud", D["aud"])


if __name__ == "__main__":
    main()
