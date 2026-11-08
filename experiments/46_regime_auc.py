"""AUC, balanced accuracy และความแม่นตามสภาวะรายสัปดาห์ (รายงานเท่านั้น ไม่ใช้เปลี่ยนโมเดล)

    python experiments/46_regime_auc.py
→ results/e16_regime/ (metrics_wf, regime, weeks, recent, recent_rows, repro_check .csv และ report.md)

ป้ายขึ้น/ไม่ขึ้น (วันนิ่ง = ไม่ขึ้น) · ทาย "ขึ้น" เมื่อ P(ขึ้น) > 0.5 · แยกรอบเช้าและบ่าย
walk-forward: อ่านไฟล์ผลทำนายเดิม (A-ud = C1235, A-un = E16, E15 MLP เฉพาะบ่าย) · E0-un = สัดส่วนวันขึ้นในข้อมูลฝึก
holdout และ 30 ต.ค.–6 พ.ย.: ฝึกใหม่รายวันด้วยข้อมูลก่อนวันทดสอบ · E15 MLP ใช้ค่าจูนรอบสุดท้าย (เริ่ม 21 ก.ย. 2026)
ไม่อ่านโมเดลที่ล็อก ไม่แก้ rules.yaml · ไม่ใช้ตัดสินโมเดล
"""
import ast
import importlib.util
import json
import pickle

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from forecast.config import OUTPUT_DIR, PROCESSED_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset
from forecast.models.e9 import target_rows
from forecast.models.nested_tune import build, proba


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s17, s30, s31, s45 = [_module(n) for n in ("17_label_variants.py", "30_e16_up_notup.py", "31_e14_finetune.py", "45_morning_daily_wf.py")]
KEYS = ["date", "symbol", "session"]
RES = ROOT / "results" / "e16_regime"
EXP = OUTPUT_DIR / "experiments"
TH = {"morning": "เช้า", "afternoon": "บ่าย"}
N_BOOT = 2000
BINS = ["< 35%", "35–50%", "50–65%", "> 65%"]


def models_of(session):
    return ["A-ud", "A-un", "E15 MLP", "E0-un"] if session == "afternoon" else ["A-ud", "A-un", "E0-un"]


def read_pred(path, tz):
    p = pd.read_csv(path)
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def bal_acc(y, p):
    if not 0 < y.sum() < len(y):
        return np.nan
    pred = p > 0.5
    return (pred[y == 1].mean() + (~pred[y == 0]).mean()) / 2


def boot_metrics(x, names, seed=0):
    """block bootstrap รายวัน: ทุกแบบใช้ชุดวันเดียวกันในแต่ละครั้ง · ครั้งที่มีป้ายเดียว → ข้าม"""
    y = (x["direction"] == 1).to_numpy(int)
    days = x["date"].to_numpy()
    groups = list(pd.Series(np.arange(len(x))).groupby(days).indices.values())
    ps = {n: x[f"p_{n}"].to_numpy(float) for n in names}
    rng = np.random.default_rng(seed)
    a = {n: [] for n in names}
    b = {n: [] for n in names}
    for _ in range(N_BOOT):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if not 0 < y[rows].sum() < len(rows):
            continue
        for n in names:
            a[n].append(roc_auc_score(y[rows], ps[n][rows]))
            b[n].append(bal_acc(y[rows], ps[n][rows]))
    a, b = {n: np.array(v) for n, v in a.items()}, {n: np.array(v) for n, v in b.items()}
    rows = []
    q = lambda v: (np.quantile(v, 0.025), np.quantile(v, 0.975))
    for n in names:
        pred = ps[n] > 0.5
        rows.append({"model": n, "n": len(x), "n_days": len(groups), "n_up": int(y.sum()), "up_share": y.mean(),
                     "auc": auc(y, ps[n]), "auc_lo": q(a[n])[0], "auc_hi": q(a[n])[1],
                     "bal_acc": bal_acc(y, ps[n]), "bal_lo": q(b[n])[0], "bal_hi": q(b[n])[1],
                     "acc": (pred == (y == 1)).mean(), "pred_up": pred.mean(), "n_boot_used": len(a[n])})
    if "A-ud" in names and "A-un" in names:
        rows.append({"model": "A-ud − A-un", "n": len(x), "n_days": len(groups), "n_up": int(y.sum()), "up_share": y.mean(),
                     "auc": auc(y, ps["A-ud"]) - auc(y, ps["A-un"]), "auc_lo": q(a["A-ud"] - a["A-un"])[0], "auc_hi": q(a["A-ud"] - a["A-un"])[1],
                     "bal_acc": bal_acc(y, ps["A-ud"]) - bal_acc(y, ps["A-un"]), "bal_lo": q(b["A-ud"] - b["A-un"])[0], "bal_hi": q(b["A-ud"] - b["A-un"])[1],
                     "acc": ((ps["A-ud"] > 0.5) == (y == 1)).mean() - ((ps["A-un"] > 0.5) == (y == 1)).mean(), "pred_up": np.nan, "n_boot_used": len(a["A-ud"])})
    return pd.DataFrame(rows)


def walk_forward(rules):
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    ud = read_pred(EXP / "C1235" / "E12_fd_lr_predictions.csv", tz)
    un = read_pred(EXP / "E16" / "a_un_predictions.csv", tz)
    e15 = read_pred(EXP / "E15" / "predictions.csv", tz)
    mlp = e15[(e15["model"] == "mlp") & (e15["split"] == "wf") & (e15["session"] == "afternoon")].rename(columns={"p": "p_up"})
    e0 = s30.e0_window(ds, rules)
    e0["date"] = pd.to_datetime(e0["date"])
    out = {}
    for sess in TH:
        parts = {"A-ud": ud, "A-un": un, "E0-un": e0}
        if sess == "afternoon":
            parts["E15 MLP"] = mlp
        x = ds[ds["session"] == sess][KEYS + ["direction"]]
        for n, p in parts.items():
            x = x.merge(p[p["session"] == sess][KEYS + ["p_up"]].rename(columns={"p_up": f"p_{n}"}), on=KEYS, validate="1:1")
        out[sess] = x[x["direction"].notna()].sort_values(KEYS).reset_index(drop=True)
    return out


def regime(x, sess):
    """สัปดาห์จันทร์–อาทิตย์ · ขอบ: < 35 | 35 ≤ x < 50 | 50 ≤ x ≤ 65 | > 65 · รู้สภาวะหลังจบสัปดาห์เท่านั้น"""
    names = models_of(sess)
    x = x.copy()
    x["week"] = x["date"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    x["up"] = (x["direction"] == 1).astype(int)
    wk = x.groupby("week").agg(n=("up", "size"), n_up=("up", "sum")).reset_index()
    wk["up_share"] = wk["n_up"] / wk["n"]
    wk["bin"] = np.select([wk["up_share"] < 0.35, wk["up_share"] < 0.5, wk["up_share"] <= 0.65], BINS[:3], BINS[3])
    x = x.merge(wk[["week", "bin"]], on="week")
    rows = []
    for b in BINS + ["ทั้งหมด"]:
        g = x if b == "ทั้งหมด" else x[x["bin"] == b]
        n_weeks = wk["week"].nunique() if b == "ทั้งหมด" else int((wk["bin"] == b).sum())
        r = {"session": TH[sess], "bin": b, "n_weeks": n_weeks, "n_rounds": len(g), "n_up": int(g["up"].sum())}
        for n in names:
            r[f"acc {n}"] = ((g[f"p_{n}"] > 0.5).astype(int) == g["up"]).mean() if len(g) else np.nan
        rows.append(r)
    return pd.DataFrame(rows), wk.assign(session=TH[sess])


def mlp_params():
    t = pd.read_csv(EXP / "E15" / "tuning.csv")
    r = t[(t["model"] == "mlp") & (t["session"] == "afternoon")].sort_values("period").iloc[-1]
    return ast.literal_eval(r["params"]), json.loads(r["features"]), r["start"]


def mlp_predict(d, feats, params, test_days, train_before):
    """ลำดับแถวเดียวกับ period_job ใน 33_e15_models.py: aligned → กรองรอบบ่าย"""
    g = d[d["session"] == "afternoon"].reset_index(drop=True)
    m, y_all = target_rows(g, "up_notup")
    y = np.full(len(g), -1)
    y[np.flatnonzero(m)] = y_all
    X = g[feats].to_numpy(float)
    out = []
    for day in test_days:
        a = np.flatnonzero((g["date"] < train_before(day)).to_numpy() & (y >= 0))
        t = np.flatnonzero((g["date"] == day).to_numpy())
        assert g["date"].iloc[a].max() < day and len(t) == 2
        model = build("mlp", params, 0).fit(X[a], y[a])
        out.append(g.loc[t, KEYS].assign(p_up=proba(model, X[t])))
    return pd.concat(out, ignore_index=True)


def recent_rows(rules, new_ds, un, fd, feats, params, test_days, train_before):
    exps = load_experiments()
    aligned, _ = s31.aligned(new_ds, exps)
    frames = {}
    for sess in TH:
        s45.SESSION = sess
        x = s45.run_all(s45.morning_rows(new_ds.reset_index(drop=True)), rules, un, fd, test_days, train_before)
        if sess == "afternoon":
            p = mlp_predict(aligned, feats, params, test_days, train_before)
            x = x.merge(p.rename(columns={"p_up": "p_E15 MLP"}), on=KEYS, validate="1:1")
        frames[sess] = x.sort_values(KEYS).reset_index(drop=True)
    return frames


def repro(wf, rules, new_ds, un, fd, feats, params, hold, new):
    rows = []
    m16 = pd.read_csv(ROOT / "results" / "e16" / "metrics.csv")
    m15 = pd.read_csv(ROOT / "results" / "e15" / "metrics.csv")
    for sess in TH:
        y = wf[sess]["direction"] == 1
        for n in models_of(sess):
            ref = m15[(m15["model"] == "mlp") & (m15["group"] == TH[sess])]["acc_up_notup"].iloc[0] if n == "E15 MLP" else \
                m16[(m16["model"] == n) & (m16["group"] == TH[sess])]["acc_up_notup"].iloc[0]
            rows.append({"check": "walk-forward acc", "session": TH[sess], "model": n, "n": len(y), "value": ((wf[sess][f"p_{n}"] > 0.5) == y).mean(), "reference": ref})
    for name, days, f in [("holdout ฝึกครั้งเดียว", hold, "holdout.csv"), ("30 ต.ค.–6 พ.ย. ฝึกครั้งเดียว", new, "new_data.csv")]:
        ref = pd.read_csv(ROOT / "results" / "e16" / f)
        first = days[0]
        frames = recent_rows(rules, new_ds, un, fd, feats, params, days, lambda _: first)
        for sess, x in frames.items():
            y = (x["direction"] == 1).to_numpy(float)
            for n in models_of(sess):
                ll = s30.losses(x[f"p_{n}"].to_numpy(), y)[0].mean()
                if n == "E15 MLP":
                    if f != "holdout.csv":
                        continue
                    r15 = pd.read_csv(ROOT / "results" / "e15" / "holdout.csv").query("model == 'mlp' and session == 'afternoon'").iloc[0]
                    rows.append({"check": f"{name} log loss", "session": TH[sess], "model": n, "n": len(x), "value": ll, "reference": r15["ll"]})
                    continue
                r = ref[(ref["model"] == n) & (ref["group"] == TH[sess])].iloc[0]
                rows.append({"check": f"{name} log loss", "session": TH[sess], "model": n, "n": len(x), "value": ll, "reference": r["logloss_un"]})
    return pd.DataFrame(rows)


def fmt_ci(v, lo, hi, signed=False):
    if pd.isna(v):
        return "–"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def show_metrics(t):
    out = t.copy()
    diff = out["model"] == "A-ud − A-un"
    out["AUC [CI]"] = [fmt_ci(r.auc, r.auc_lo, r.auc_hi, d) for r, d in zip(out.itertuples(), diff)]
    out["balanced acc [CI]"] = [fmt_ci(r.bal_acc, r.bal_lo, r.bal_hi, d) for r, d in zip(out.itertuples(), diff)]
    out["แม่น"] = [f"{r.acc * 100:+.1f} จุด" if d else f"{r.acc:.1%}" for r, d in zip(out.itertuples(), diff)]
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    pd.set_option("display.width", 260, "display.max_columns", 40)
    with open(PROCESSED_DIR / "dataset_yfinance_60m_pcoff_atc_c1235_new.pkl", "rb") as f:
        new_ds = pickle.load(f).reset_index(drop=True)
    days = sorted(new_ds.loc[new_ds["session"] == "morning", "date"].unique())
    new, hold = days[-6:], days[-16:-6]
    assert str(new[0])[:10] == "2026-10-30" and str(hold[0])[:10] == "2026-10-14" and str(new[-1])[:10] == "2026-11-06"
    un, fd, _ = s30.variants()
    params, feats, tune_start = mlp_params()
    assert tune_start < "2026-10-14", tune_start

    wf = walk_forward(rules)
    check = repro(wf, rules, new_ds, un, fd, feats, params, hold, new)
    check["diff"] = check["value"] - check["reference"]
    check.to_csv(RES / "repro_check.csv", index=False, encoding="utf-8-sig")
    print(check.round(4).to_string(index=False), flush=True)

    tab = pd.concat([boot_metrics(wf[s], models_of(s)).assign(session=TH[s], first=wf[s]["date"].min().date(), last=wf[s]["date"].max().date()) for s in TH])
    tab.to_csv(RES / "metrics_wf.csv", index=False, encoding="utf-8-sig")
    reg, weeks = zip(*[regime(wf[s], s) for s in TH])
    reg, weeks = pd.concat(reg), pd.concat(weeks)
    reg.to_csv(RES / "regime.csv", index=False, encoding="utf-8-sig")
    weeks.to_csv(RES / "weeks.csv", index=False, encoding="utf-8-sig")

    test_days = hold + new
    rec = recent_rows(rules, new_ds, un, fd, feats, params, test_days, lambda day: day)
    old = pd.read_csv(ROOT / "results" / "e16_morning_daily" / "rows.csv")
    am = rec["morning"].assign(d=lambda t: t["date"].dt.strftime("%Y-%m-%d")).merge(old, left_on=["d", "symbol"], right_on=["date", "symbol"], suffixes=("", "_old"))
    assert len(am) == 32
    print("ผลต่างสูงสุดจาก rows.csv รอบเช้า:", {n: float((am[f"p_{n}"] - am[f"p_{n}_old"]).abs().max()) for n in ["A-un", "A-ud", "E0-un"]}, flush=True)

    periods = [("14–29 ต.ค. (holdout)", hold), ("30 ต.ค.–6 พ.ย.", new), ("รวม 14 ต.ค.–6 พ.ย.", test_days)]
    recent = []
    rows_out = []
    for s, x in rec.items():
        out = x.drop(columns=[c for c in ["p_start", "p_end", "trained_until"] if c in x]).assign(session=TH[s])
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        rows_out.append(out)
        for label, ds_ in periods:
            g = x[x["date"].isin(ds_)].reset_index(drop=True)
            recent.append(boot_metrics(g, models_of(s)).assign(session=TH[s], period=label))
    recent = pd.concat(recent)
    recent.to_csv(RES / "recent.csv", index=False, encoding="utf-8-sig")
    pd.concat(rows_out).to_csv(RES / "recent_rows.csv", index=False, encoding="utf-8-sig")

    cols = ["session", "model", "n", "n_up", "AUC [CI]", "balanced acc [CI]", "แม่น"]
    t1, t3 = show_metrics(tab)[cols], show_metrics(recent)[["period"] + cols]
    r2 = reg.copy()
    for c in [c for c in r2 if c.startswith("acc ")]:
        r2[c] = r2[c].map(lambda v: "–" if pd.isna(v) else f"{v:.1%}")
    (RES / "report.md").write_text(
        "# AUC, balanced accuracy และสภาวะรายสัปดาห์ (รายงานเท่านั้น)\n\n`experiments/46_regime_auc.py` · CI 95% จาก block bootstrap รายวัน 2,000 ครั้ง\n\n"
        "## 1. Walk-forward\n\n" + s45.md_table(t1) + "\n## 2. ตามสภาวะรายสัปดาห์\n\n" + s45.md_table(r2.fillna("–")) +
        "\n## 3. holdout และ 30 ต.ค.–6 พ.ย. (รายงานเท่านั้น)\n\n" + s45.md_table(t3), encoding="utf-8")
    print(tab[["session", "first", "last"]].drop_duplicates().to_string(index=False))
    print(t1.to_string(index=False))
    print(r2.to_string(index=False))
    print(t3.to_string(index=False))
    print(recent[["period", "session", "model", "n_boot_used", "pred_up"]].to_string(index=False))


if __name__ == "__main__":
    main()
