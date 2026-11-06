"""E14 (รอบ 7 ข้อ 5): D และ D+C — เกณฑ์ลงทะเบียนใน results/e14_preregistration.md

    python experiments/31_e14_finetune.py [--one]      # --one = วัดเวลา 1 รอบ
→ outputs/experiments/E14/*_predictions.csv, results/e14/*.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util
import sys
import time

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import tick_size
from forecast.evaluation.runner import load_or_build_dataset
from forecast.evaluation.splits import holdout_split, walk_forward_splits
from forecast.models.finetune_lr import MIN_DAYS, PretrainFineTune, logloss


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s17, s18 = _module("17_label_variants.py"), _module("18_score_round4.py")
KEYS = ["date", "symbol", "session"]
OUT, RES = OUTPUT_DIR / "experiments" / "E14", ROOT / "results" / "e14"
PM_ONLY = ["f_gap_open", "f_rv_intra"]
Z = 2.5  # ไม่ใช้ · CI คำนวณจาก quantile ของ bootstrap


def aligned(ds, exps):
    """→ ds + t_r_prev1, t_r_prev2, dummy · shared = feature รวมสองรอบ"""
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")["features"]
    d = ds.sort_values(["symbol", "date", "session"]).copy()
    am = d["session"] == "morning"
    prev_am = d[am].groupby("symbol")["r"].shift(1)
    d["t_r_prev1"] = np.where(am, d["f_r_prev_afternoon"], d["f_r_open_to_cut"])
    d["t_r_prev2"] = d["f_r_prev_afternoon"]
    d.loc[am, "t_r_prev2"] = prev_am
    d["d_sess_pm"] = (~am).astype(float)
    d["d_sym_advanc"] = (d["symbol"] == "ADVANC").astype(float)
    d["tick1"] = np.isclose(tick_size(d["p_ref"].to_numpy(float)), 1.0)
    drop = {"f_r_prev_afternoon", "f_r_open_to_cut", *PM_ONLY}
    shared = [c for c in fd["morning"] if c not in drop] + ["t_r_prev1", "t_r_prev2"]
    return d.reset_index(drop=True), shared


def part(g):
    return {"X": g, "y": (g["direction"] == 1).to_numpy(int), "dates": g["date"].to_numpy()}


def block(d, shared, session, kind, test_dates):
    """ฝึกจากวันก่อนบล็อกทดสอบเท่านั้น → (แถวผล, κ, inner log loss)"""
    start = np.min(test_dates)
    tr = d[(d["date"].to_numpy() < start) & d["direction"].notna()]
    ses = tr[tr["session"] == session]
    pre_cols = shared + ["d_sess_pm", "d_sym_advanc"]
    ft_cols = shared + ["d_sym_advanc"] + (PM_ONLY if session == "afternoon" else [])
    const = {"d_sess_pm": 1.0 if session == "afternoon" else 0.0}
    ft = ses
    if kind == "D+C":
        ft = ses[ses["tick1"]]
        if ft[ft["symbol"] == "KBANK"]["date"].nunique() < MIN_DAYS:   # ขั้นต่ำนับจากแถว KBANK
            ft = ft.iloc[:0]
    m = PretrainFineTune()
    if len(ft) == 0:
        m.ft_days_min = 10 ** 9
        m.fit(part(tr), part(ses), pre_cols, ft_cols, const, inner_val=part(ses))
    else:
        m.fit(part(tr), part(ft), pre_cols, ft_cols, const, inner_val=part(ses))
    te = d[(d["session"] == session) & d["date"].isin(test_dates)]
    out = te[KEYS + ["direction", "p_ref", "tick1"]].copy()
    out["p_up"] = m.predict(te)
    out["kappa"] = m.kappa
    out["inner_ll"] = m.inner_ll
    return out


def run(d, shared, rules, kind, one=False, holdout=False):
    ev = rules["evaluation"]
    jobs = []
    for session in ["morning", "afternoon"]:
        g = d[d["session"] == session]
        dates = g["date"].to_numpy()
        splits = (holdout_split(dates, ev["holdout_days"]) if holdout else
                  walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]))
        jobs += [(session, te) for _, te in splits]
    if one:
        jobs = jobs[-1:]
    res = Parallel(n_jobs=-1, verbose=0)(delayed(block)(d, shared, s, kind, te) for s, te in jobs)
    return pd.concat(res, ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def boot_ci(v, dates, conf, n_boot, seed=0):
    df = pd.DataFrame({"v": np.asarray(v, float), "d": np.asarray(dates)}).dropna()
    s = df.groupby("d")["v"].agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    b = s["sum"].to_numpy()[idx].sum(1) / s["count"].to_numpy()[idx].sum(1)
    a = (1 - conf) / 2
    return s["sum"].sum() / s["count"].sum(), np.quantile(b, a), np.quantile(b, 1 - a)


def compare(d, a, b, conf, n_boot, mask=None):
    """ผลต่าง a − b (log loss, ความแม่น) แยกรอบ"""
    y = (d["direction"] == 1).to_numpy(float)
    rows = []
    for label, m in [("เช้า", d["session"] == "morning"), ("บ่าย", d["session"] == "afternoon"),
                     ("KBANK เช้า", (d["session"] == "morning") & (d["symbol"] == "KBANK")),
                     ("KBANK บ่าย", (d["session"] == "afternoon") & (d["symbol"] == "KBANK")),
                     ("ADVANC เช้า", (d["session"] == "morning") & (d["symbol"] == "ADVANC")),
                     ("ADVANC บ่าย", (d["session"] == "afternoon") & (d["symbol"] == "ADVANC"))]:
        m = m.to_numpy() & (mask if mask is not None else True)
        if not m.any():
            continue
        pa, pb = d[f"p_{a}"].to_numpy()[m], d[f"p_{b}"].to_numpy()[m]
        la, lb = logloss(y[m], pa), logloss(y[m], pb)
        oa, ob = ((pa > .5) == (y[m] == 1)).astype(float), ((pb > .5) == (y[m] == 1)).astype(float)
        dates = d["date"].to_numpy()[m]
        r = {"a": a, "b": b, "group": label, "n": int(m.sum()), "acc_a": oa.mean(), "acc_b": ob.mean(), "ll_a": la.mean(), "ll_b": lb.mean(),
             "pred_up_a": (pa > .5).mean()}
        r.update(zip(["d_ll", "d_ll_lo", "d_ll_hi"], boot_ci(la - lb, dates, conf, n_boot)))
        r.update(zip(["d_acc", "d_acc_lo", "d_acc_hi"], boot_ci(oa - ob, dates, conf, n_boot)))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    one = "--one" in sys.argv
    rules = load_rules()
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    d, shared = aligned(ds, load_experiments())
    print(f"{len(d)} แถว · shared {len(shared)} feature", flush=True)
    if one:
        t = time.perf_counter()
        r = run(d, shared, rules, "D", one=True)
        print(f"1 บล็อก: {time.perf_counter() - t:.1f} วิ · κ {r['kappa'].iat[0]} · inner {r['inner_ll'].iat[0]:.4f}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    pred = {}
    for kind, name in [("D", "d"), ("D+C", "dc")]:
        t = time.perf_counter()
        pred[kind] = run(d, shared, rules, kind)
        pred[kind].to_csv(OUT / f"{name}_predictions.csv", index=False)
        print(f"{kind}: {len(pred[kind])} แถว · {pred[kind]['date'].max().date()} · {time.perf_counter() - t:.0f} วิ", flush=True)
    assert all(str(p["date"].max())[:10] == "2026-10-12" for p in pred.values())
    a_un = pd.read_csv(OUTPUT_DIR / "experiments" / "E16" / "a_un_predictions.csv")
    a_ud = s18.load("C1235", "E12_fd_lr", tz)
    e0 = pd.read_csv(ROOT / "results" / "e16" / "metrics.csv")  # มีไว้ตรวจว่า E16 รันแล้ว
    assert len(e0)
    a_un["date"] = pd.to_datetime(a_un["date"], utc=True).dt.tz_convert(tz)
    s30 = _module("30_e16_up_notup.py")
    e0_un = s30.e0_window(ds, rules)
    x = pred["D"][KEYS + ["direction", "tick1", "p_up"]].rename(columns={"p_up": "p_D"})
    for n, p in [("D+C", pred["D+C"]), ("A-un", a_un), ("A-ud", a_ud), ("E0-un", e0_un)]:
        x = x.merge(p[KEYS + ["p_up"]].rename(columns={"p_up": f"p_{n}"}), on=KEYS)
    x = x[x["direction"].notna()].reset_index(drop=True)
    print(f"เทียบบน {len(x)} รอบ", flush=True)
    n_boot, conf = rules["evaluation"]["n_boot"], 0.9875
    rows = []
    tick = x["tick1"].to_numpy()
    for a in ["D", "D+C"]:
        for b in ["A-un", "E0-un", "A-ud"]:
            rows.append(compare(x, a, b, conf, n_boot).assign(ช่วง="ทั้งช่วง"))
            rows.append(compare(x, a, b, conf, n_boot, tick).assign(ช่วง="ช่องราคา 1 บาท"))
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(RES / "compare.csv", index=False, encoding="utf-8-sig")

    # เกณฑ์ลงทะเบียน: เทียบ A-un แยกรอบ
    v = []
    get = lambda a, s, g: t[(t["a"] == a) & (t["b"] == "A-un") & (t["ช่วง"] == s) & (t["group"] == g)].iloc[0]
    for g in ["เช้า", "บ่าย"]:
        for a, win, other in [("D", "ทั้งช่วง", "ช่องราคา 1 บาท"), ("D+C", "ช่องราคา 1 บาท", "ทั้งช่วง")]:
            w, o = get(a, win, g), get(a, other, g)
            ok = bool(w["d_ll_hi"] < 0 and w["d_acc_hi"] >= 0 and o["d_ll_lo"] <= 0)
            v.append({"รอบ": g, "แบบ": a, "ช่วงชนะ": win, "d_ll": w["d_ll"], "lo": w["d_ll_lo"], "hi": w["d_ll_hi"], "d_acc": w["d_acc"],
                      "acc_lo": w["d_acc_lo"], "acc_hi": w["d_acc_hi"], "อีกช่วง d_ll_lo": o["d_ll_lo"], "ผ่าน": ok})
    v = pd.DataFrame(v)
    v.to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")

    # ตัวแทนเข้าข้อ 9: inner-CV log loss เฉลี่ยรายบล็อก ไม่ใช้ผลชุดทดสอบ
    rep = []
    for sess in ["morning", "afternoon"]:
        s = {k: pred[k][pred[k]["session"] == sess].groupby("date")["inner_ll"].first().mean() for k in pred}
        kap = {k: pred[k][pred[k]["session"] == sess].drop_duplicates("date")["kappa"].value_counts().to_dict() for k in pred}
        rep.append({"ช่อง": f"{sess}_dir", "inner_D": s["D"], "inner_D+C": s["D+C"], "ตัวแทน": min(s, key=s.get), "κ D": kap["D"], "κ D+C": kap["D+C"]})
    rep = pd.DataFrame(rep)
    rep.to_csv(RES / "representatives.csv", index=False, encoding="utf-8-sig")

    k = ds[ds["symbol"] == "KBANK"].assign(tick1=lambda f: np.isclose(tick_size(f["p_ref"].to_numpy(float)), 1.0))
    fl = k[k["direction"].notna()].groupby(["session", "tick1"])["direction"].agg(n="size", flat=lambda s: (s == 0).mean())
    fl.to_csv(RES / "kbank_flat_rate.csv", encoding="utf-8-sig")

    h = {kk: run(d, shared, rules, kk, holdout=True) for kk in ["D", "D+C"]}
    y = pd.concat([h[kk].assign(model=kk) for kk in h])
    y["ok"] = (y["p_up"] > .5) == (y["direction"] == 1)
    y = y[y["direction"].notna()]
    y["ll"] = logloss((y["direction"] == 1).to_numpy(float), y["p_up"].to_numpy())
    ho = y.groupby(["model", "session"]).agg(n=("ok", "size"), acc=("ok", "mean"), ll=("ll", "mean"), pred_up=("p_up", lambda s: (s > .5).mean())).reset_index()
    ho.to_csv(RES / "holdout.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    show = ["a", "b", "ช่วง", "group", "n", "acc_a", "acc_b", "pred_up_a", "d_ll", "d_ll_lo", "d_ll_hi", "d_acc", "d_acc_lo", "d_acc_hi"]
    print(t[t["group"].isin(["เช้า", "บ่าย"])][show].round(4).to_string(index=False))
    print("\nคำตัดสิน (เทียบ A-un, CI 98.75%)\n", v.round(4).to_string(index=False))
    print("\nตัวแทน\n", rep.round(5).to_string(index=False))
    print("\nKBANK วันนิ่ง\n", fl.round(3).to_string())
    print("\nholdout (รายงานเฉย ๆ)\n", ho.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
