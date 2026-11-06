"""E16 (รอบ 7 ข้อ 3): เป้าหมายขึ้น/ไม่ขึ้น — เกณฑ์ลงทะเบียนใน results/e16_preregistration.md

    python experiments/30_e16_up_notup.py
→ outputs/experiments/E16/*_predictions.csv, results/e16_report.md, results/e16/*.csv
"""
import copy
import importlib.util

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.metrics import tick_size
from forecast.evaluation.runner import load_or_build_dataset, run_variant
from forecast.evaluation.splits import walk_forward_splits


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s17, s18 = _module("17_label_variants.py"), _module("18_score_round4.py")
KEYS = ["date", "symbol", "session"]
OUT, RES = OUTPUT_DIR / "experiments" / "E16", ROOT / "results" / "e16"
GROUPS = [("ALL", None, None), ("เช้า", None, "morning"), ("บ่าย", None, "afternoon"),
          ("KBANK เช้า", "KBANK", "morning"), ("KBANK บ่าย", "KBANK", "afternoon"),
          ("ADVANC เช้า", "ADVANC", "morning"), ("ADVANC บ่าย", "ADVANC", "afternoon")]
MODELS = ["A-un", "A-dec", "A-ud", "E0-un", "E0-un-120", "E0-ud"]


def variants():
    exps = load_experiments()
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")
    un = {**copy.deepcopy(fd), "name": "a_un", "params": {**fd.get("params", {}), "target": "up_notup"}}
    adir = next(v for v in exps["E9FLAT"]["variants"] if v["name"] == "adir_lr")
    return un, fd, adir


def e0_window(ds, rules, window=None):
    """E0-un: สัดส่วนวันขึ้นของข้อมูลฝึก แยกหุ้นและรอบ (window = เฉพาะ n วันล่าสุด) ใช้ splits เดียวกับ walk-forward"""
    ev, out = rules["evaluation"], []
    for _, g in ds.groupby(["symbol", "session"]):
        dates = g["date"].to_numpy()
        for tr, te in walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]):
            tr = tr[-window:] if window else tr
            y = g.loc[np.isin(dates, tr) & g["direction"].notna().to_numpy(), "direction"] == 1
            out.append(g.loc[np.isin(dates, te), KEYS].assign(p_up=float(y.mean())))
    return pd.concat(out, ignore_index=True)


def losses(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), (p - y) ** 2, ((p > 0.5) == (y == 1)).astype(float)


def table(d, n_boot):
    """d: แถวร่วม มี p_<แบบ> → ตารางแบบ × กลุ่ม (3 กติกา, log loss, Brier, ผลต่างเทียบ E0-un พร้อม CI)"""
    y = (d["direction"] == 1).to_numpy(float)
    flat = (d["direction"] == 0).to_numpy()
    rows, base = [], losses(d["p_E0-un"].to_numpy(), y)
    for name in MODELS:
        ll, br, ok = losses(d[f"p_{name}"].to_numpy(), y)
        pred_up = d[f"p_{name}"].to_numpy() > 0.5
        for label, sym, sess in GROUPS:
            m = np.ones(len(d), bool)
            if sym:
                m &= (d["symbol"] == sym).to_numpy()
            if sess:
                m &= (d["session"] == sess).to_numpy()
            if not m.any():
                continue
            r = {"model": name, "group": label, "n": int(m.sum()), "n_days": d.loc[m, "date"].nunique(),
                 "up_rate": y[m].mean(), "flat_rate": flat[m].mean(), "pred_up_rate": pred_up[m].mean(),
                 "acc_up_notup": ok[m].mean(), "logloss_un": ll[m].mean(), "brier_un": br[m].mean(),
                 "acc_flat_excluded": ok[m & ~flat].mean() if (m & ~flat).any() else np.nan,
                 "acc_flat_wrong": (ok[m] * ~flat[m]).mean()}
            if name != "E0-un":
                dates = d.loc[m, "date"]
                r.update(zip(["d_ll", "d_ll_lo", "d_ll_hi"], s18.boot_mean(ll[m] - base[0][m], dates, n_boot)))
                r.update(zip(["d_acc", "d_acc_lo", "d_acc_hi"], s18.boot_mean(ok[m] - base[2][m], dates, n_boot)))
                r.update(zip(["d_brier", "d_brier_lo", "d_brier_hi"], s18.boot_mean(br[m] - base[1][m], dates, n_boot)))
            rows.append(r)
    return pd.DataFrame(rows)


def pairs(d, n_boot):
    y = (d["direction"] == 1).to_numpy(float)
    rows = []
    for a, b in [("A-un", "A-ud"), ("A-dec", "A-un")]:
        la, _, oa = losses(d[f"p_{a}"].to_numpy(), y)
        lb, _, ob = losses(d[f"p_{b}"].to_numpy(), y)
        for label, sym, sess in GROUPS:
            m = np.ones(len(d), bool)
            if sym:
                m &= (d["symbol"] == sym).to_numpy()
            if sess:
                m &= (d["session"] == sess).to_numpy()
            dates = d.loc[m, "date"]
            r = {"pair": f"{a} − {b}", "group": label}
            r.update(zip(["d_ll", "d_ll_lo", "d_ll_hi"], s18.boot_mean(la[m] - lb[m], dates, n_boot)))
            r.update(zip(["d_acc", "d_acc_lo", "d_acc_hi"], s18.boot_mean(oa[m] - ob[m], dates, n_boot)))
            rows.append(r)
    return pd.DataFrame(rows)


def importance(pred, name):
    """permutation importance: log loss บนป้ายขึ้น/ไม่ขึ้นเพิ่มเท่าไรเมื่อสลับ feature นั้น"""
    d = pred[pred["direction"].notna()]
    y = (d["direction"] == 1).to_numpy(float)
    base = losses(d["p_up"].to_numpy(), y)[0].mean()
    rows = []
    for c in [c for c in d if c.startswith("perm_")]:
        ok = d[c].notna().to_numpy()
        if ok.any():
            rows.append({"model": name, "feature": c[5:], "d_logloss": losses(d[c].to_numpy()[ok], y[ok])[0].mean() - base})
    return pd.DataFrame(rows).sort_values("d_logloss", ascending=False)


def join(parts):
    keys = None
    for p in parts.values():
        k = p[KEYS].astype({"date": "datetime64[ns, Asia/Bangkok]"})
        keys = k if keys is None else keys.merge(k, on=KEYS)
    out = keys
    for n, p in parts.items():
        cols = KEYS + [c for c in ["p_up", "direction", "p_ref"] if c in p]
        x = p[cols].astype({"date": "datetime64[ns, Asia/Bangkok]"}).rename(columns={"p_up": f"p_{n}"})
        if "direction" in out:
            x = x.drop(columns=[c for c in ["direction", "p_ref"] if c in x])
        out = out.merge(x, on=KEYS)
    return out[out["direction"].notna()].reset_index(drop=True)


def verdict(tab):
    rows = []
    for sess, label in [("morning", "เช้า"), ("afternoon", "บ่าย")]:
        cand = []
        for name in ["A-un", "A-dec"]:
            r = tab[(tab["model"] == name) & (tab["group"] == label)].iloc[0]
            ok = bool(r["d_ll_hi"] < 0 and r["d_acc_hi"] >= 0)
            rows.append({"รอบ": label, "แบบ": name, "d_ll": r["d_ll"], "d_ll_lo": r["d_ll_lo"], "d_ll_hi": r["d_ll_hi"],
                         "d_acc": r["d_acc"], "d_acc_lo": r["d_acc_lo"], "d_acc_hi": r["d_acc_hi"], "ผ่าน": ok})
            if ok:
                cand.append((r["logloss_un"], name))
        rows[-1]["เลือก"] = min(cand)[1] if cand else "A-ud (ไม่ผ่าน)"
        rows[-2]["เลือก"] = rows[-1]["เลือก"]
    return pd.DataFrame(rows)


def report_only(ds, rules, days, un, fd, adir):
    """holdout / ข้อมูลใหม่: ฝึกทุกวันก่อนช่วงทดสอบ days วันท้าย — รายงานเฉย ๆ"""
    r = copy.deepcopy(rules)
    r["evaluation"]["holdout_days"] = days
    un2, fd2 = {**un, "params": {**un["params"], "permutation": False}}, {**fd, "params": {**fd.get("params", {}), "permutation": False}}
    a, u, f = [run_variant(ds, v, r, holdout_eval=True) for v in (un2, fd2, adir)]
    dec = u[KEYS].merge(f[KEYS + ["p_up"]].rename(columns={"p_up": "pf"}), on=KEYS).merge(u[KEYS + ["p_up"]], on=KEYS)
    dec["p_up"] = (1 - dec["pf"]) * dec["p_up"]
    tr = ds[~ds["date"].isin(np.sort(ds["date"].unique())[-days:])]
    share = tr[tr["direction"].notna()].groupby(["symbol", "session"])["direction"].apply(lambda s: (s == 1).mean())
    e0 = a[KEYS].copy()
    e0["p_up"] = [share[(s, x)] for s, x in zip(e0["symbol"], e0["session"])]
    e0u = run_variant(ds, {"name": "b", "model": "baseline", "params": {"target": "updown"}}, r, holdout_eval=True)
    d = join({"A-un": a, "A-dec": dec, "A-ud": u, "E0-un": e0, "E0-un-120": e0, "E0-ud": e0u})
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    assert rules["label"]["price_1630"] == "closing_auction" and rules["sessions"]["afternoon"]["cutoff"] == "12:35"
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    days = np.sort(ds["date"].unique())
    assert str(days[-1])[:10] == "2026-10-29" and str(days[-10])[:10] == "2026-10-14" and str(days[-11])[:10] == "2026-10-12", days[-11:]
    n_boot = rules["evaluation"]["n_boot"]
    un, fd, adir = variants()

    a_un = run_variant(ds, un, rules)
    a_un.to_csv(OUT / "a_un_predictions.csv", index=False)
    print("A-un", len(a_un), a_un["date"].min().date(), a_un["date"].max().date(), flush=True)
    assert str(a_un["date"].max())[:10] == "2026-10-12"
    p_flat = run_variant(ds, adir, rules)
    p_flat.to_csv(OUT / "adir_lr_predictions.csv", index=False)
    print("adir_lr", len(p_flat), flush=True)

    a_ud = s18.load("C1235", "E12_fd_lr", tz)
    e0_ud = s18.load("C1235", "E12_baseline_ud", tz)
    dec = a_ud[KEYS].merge(p_flat[KEYS + ["p_up"]].rename(columns={"p_up": "pf"}), on=KEYS).merge(a_ud[KEYS + ["p_up"]], on=KEYS)
    dec["p_up"] = (1 - dec["pf"]) * dec["p_up"]
    dec[KEYS + ["p_up"]].to_csv(OUT / "a_dec_predictions.csv", index=False)
    e0, e0_120 = e0_window(ds, rules), e0_window(ds, rules, 120)
    d = join({"A-un": a_un, "A-dec": dec, "A-ud": a_ud, "E0-un": e0, "E0-un-120": e0_120, "E0-ud": e0_ud})
    d["tick1"] = np.isclose(tick_size(d["p_ref"].to_numpy()), 1.0) if "p_ref" in d else False
    print(f"เทียบบน {len(d)} รอบ · {d['date'].min().date()} – {d['date'].max().date()}", flush=True)

    tab = table(d, n_boot)
    tab.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    tick = table(d[d["tick1"]], n_boot) if d["tick1"].any() else pd.DataFrame()
    tick.to_csv(RES / "metrics_tick1.csv", index=False, encoding="utf-8-sig")
    pr = pairs(d, n_boot)
    pr.to_csv(RES / "pairs.csv", index=False, encoding="utf-8-sig")
    v = verdict(tab)
    v.to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")
    imp = pd.concat([importance(a_un, "A-un"), importance(a_ud, "A-ud")])
    imp.to_csv(RES / "importance.csv", index=False, encoding="utf-8-sig")

    new_ds = load_or_build_dataset(rules, tag="_new", rebuild=True)
    nd = np.sort(new_ds["date"].unique())
    assert str(nd[-6])[:10] == "2026-10-30" and str(nd[-1])[:10] == "2026-11-06", nd[-7:]
    ho = table(report_only(ds, rules, 10, un, fd, adir), n_boot)
    ho.to_csv(RES / "holdout.csv", index=False, encoding="utf-8-sig")
    nw = table(report_only(new_ds, rules, 6, un, fd, adir), n_boot)
    nw.to_csv(RES / "new_data.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    show = ["model", "group", "n", "up_rate", "pred_up_rate", "acc_up_notup", "logloss_un", "d_ll", "d_ll_lo", "d_ll_hi", "d_acc", "d_acc_lo", "d_acc_hi"]
    print(tab[show].round(4).to_string(index=False))
    print("\nคำตัดสิน\n", v.round(4).to_string(index=False))
    print("\nคู่\n", pr.round(4).to_string(index=False))
    print("\nแถวช่องราคา 1 บาท\n", tick[show].round(4).to_string(index=False) if len(tick) else "ไม่มี")
    print("\nholdout (รายงานเฉย ๆ)\n", ho[ho["group"].isin(["ALL", "เช้า", "บ่าย"])][show].round(4).to_string(index=False))
    print("\n30 ต.ค.–6 พ.ย. (รายงานเฉย ๆ)\n", nw[nw["group"].isin(["ALL", "เช้า", "บ่าย"])][show].round(4).to_string(index=False))
    print("\nimportance\n", imp.groupby("model").head(6).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
