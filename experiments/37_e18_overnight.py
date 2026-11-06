"""E18: H1 ตัวตรวจระบบ → H2 ราคาเช้า → H3 ทิศทางเช้า — เกณฑ์ใน results/e18_preregistration.md · ไม่แตะระบบ live

    python experiments/37_e18_overnight.py h1     # ตรวจระบบก่อน ไม่ผ่านห้ามไปต่อ
    python experiments/37_e18_overnight.py full [--bug-check-done]   # H2 + H3 · H1 ไม่ผ่านต้องตรวจบั๊กก่อน
→ outputs/experiments/E18/*.csv, results/e18/*.csv
"""
import copy
import importlib.util
import sys
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from forecast.config import OUTPUT_DIR, RAW_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset, run_variant
from forecast.evaluation.splits import walk_forward_splits
from forecast.features.overnight import FEATURES, build_features

warnings.filterwarnings("ignore")


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s17, s18, s31, s32 = [_module(n) for n in ["17_label_variants.py", "18_score_round4.py", "31_e14_finetune.py", "32_e17a_price.py"]]
EXT = RAW_DIR / "external"
KEYS = ["date", "symbol", "session"]
RES, OUT = ROOT / "results" / "e18", OUTPUT_DIR / "experiments" / "E18"


def load_tables():
    names = {"^GSPC": "GSPC", "THD": "THD", "^N225": "N225", "^KS11": "KS11", "THB=X": "THB_X", "^SET50": "SET50"}
    return {t: pd.read_csv(EXT / f"{n}_1d.csv", index_col=0, parse_dates=True) for t, n in names.items()}, pd.read_csv(EXT / "THB_60m.csv", index_col=0)


def dataset(rules):
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(rules["timezone"]) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    tables, hourly = load_tables()
    days = pd.DatetimeIndex(np.sort(ds["date"].unique()))
    fx = build_features(days, tables, hourly)
    d = ds.merge(fx, on="date", how="left")
    pm = d["session"] == "afternoon"
    d.loc[pm, FEATURES] = np.nan
    d["fxa_spx"], d["fxa_n225"] = d["fx_spx"].abs(), d["fx_n225_gap"].abs()
    return d, fx


def boot_corr(a, b, dates, n=2000, seed=0):
    df = pd.DataFrame({"a": a, "b": b, "d": dates}).dropna()
    groups = [g[["a", "b"]].to_numpy() for _, g in df.groupby("d")]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        out.append(np.corrcoef(x.T)[0, 1])
    return np.corrcoef(df["a"], df["b"])[0, 1], np.quantile(out, .025), np.quantile(out, .975), len(df)


def h1(d, rules):
    am = d[(d["session"] == "morning") & d["p_start"].notna() & d["p_ref"].notna()].copy()
    am["gap"] = np.log(am["p_start"] / am["p_ref"])
    ev = rules["evaluation"]
    rows, coefs = [], []
    for sym, g in am.groupby("symbol"):
        g = g.sort_values("date")
        dates = g["date"].to_numpy()
        pred = pd.Series(np.nan, index=g.index)
        for tr, te in walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]):
            a, b = np.isin(dates, tr), np.isin(dates, te)
            med = g.loc[a, FEATURES].median()
            m = LinearRegression().fit(g.loc[a, FEATURES].fillna(med), g.loc[a, "gap"])
            pred[b] = m.predict(g.loc[b, FEATURES].fillna(med))
            coefs.append(dict(zip(FEATURES, m.coef_), symbol=sym))
        r, lo, hi, n = boot_corr(pred.to_numpy(), g["gap"].to_numpy(), dates, ev["n_boot"])
        rows.append({"symbol": sym, "n": n, "corr": r, "lo": lo, "hi": hi, "share_gap_up": (g["gap"] > 0).mean()})
    t = pd.DataFrame(rows)
    c = pd.DataFrame(coefs).groupby("symbol").mean()
    pos = {f: bool((c[f] > 0).all()) for f in ["fx_spx", "fx_n225_gap", "fx_thd_prem"]}
    ok = bool((t["lo"] > 0).all() and all(pos.values()))
    RES.mkdir(parents=True, exist_ok=True)
    t.to_csv(RES / "h1.csv", index=False, encoding="utf-8-sig")
    c.to_csv(RES / "h1_coef.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print(c.round(4).to_string())
    print("ทิศสัมประสิทธิ์ตามคาด", pos, "· H1", "ผ่าน" if ok else "ไม่ผ่าน → หยุดและหาบั๊ก")
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "h1"
    rules = load_rules()
    d, fx = dataset(rules)
    print(f"{len(d)} แถว · feature ว่าง (รอบเช้า): {d[d['session'] == 'morning'][FEATURES].isna().mean().round(4).to_dict()}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    fx.isna().sum().to_frame("n_missing").to_csv(RES / "v9_missing.csv", encoding="utf-8-sig")
    ok = h1(d, rules)
    if mode == "h1":
        return
    if not ok and "--bug-check-done" not in sys.argv:     # ต้องตรวจบั๊กและบันทึกใน results/e18_report.md ก่อน
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    exps = load_experiments()
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    eq = {**copy.deepcopy(v["aprice_lr"]), "features": list(v["aprice_lr"]["features"]) + ["fxa_spx", "fxa_n225"]}
    f = v["pprice_lr"]["features"]
    up = {**copy.deepcopy(v["pprice_lr"]), "sessions": ["morning"], "features": {"morning": list(f["morning"]) + FEATURES, "afternoon": list(f["afternoon"])}}
    p_eq = run_variant(d, eq, rules)
    p_up = run_variant(d, up, rules)
    p_eq.to_csv(OUT / "aprice_predictions.csv", index=False)
    p_up.to_csv(OUT / "pprice_predictions.csv", index=False)

    base_eq, base_up = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr"]]
    x = d[d["session"] == "morning"][KEYS + ["p_ref", "p_end"]]
    x = x.merge(base_eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(base_up[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS)
    x = x.merge(p_eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_B"}), on=KEYS).merge(p_up[KEYS + ["p_up"]].rename(columns={"p_up": "up_B"}), on=KEYS)
    x = x[x["p_end"].notna() & x["p_ref"].notna()].reset_index(drop=True)
    pref = x["p_ref"].to_numpy(float)
    x["tick1"] = False
    x["pt_PR-A"] = s18.price_rule(pref, x["eq_A"].to_numpy(), x["up_A"].to_numpy())[0]
    x["pt_E18"] = s18.price_rule(pref, x["eq_B"].to_numpy(), x["up_B"].to_numpy())[0]
    x["pt_P(c)"] = pref
    s32.CONF = 0.95
    pr = s32.score(x, ["PR-A", "E18", "P(c)"], n_boot)
    pr = pr[pr["ช่วง"] == "ทั้งช่วง"]
    pr.to_csv(RES / "h2_price.csv", index=False, encoding="utf-8-sig")
    g = pr[pr["model"] == "E18"]
    allg = g[g["group"] == "เช้า"].iloc[0]
    stocks = g[g["group"].str.contains("KBANK|ADVANC")]
    h2 = bool(allg["d_mae_hi"] < 0 and (stocks["d_mae_lo"] <= 0).all())

    fd = next(x_ for x_ in exps["E12"]["variants"] if x_["name"] == "fd_lr")
    un = {**copy.deepcopy(fd), "name": "e18_un", "sessions": ["morning"], "params": {**fd.get("params", {}), "target": "up_notup", "permutation": False},
          "features": {"morning": list(fd["features"]["morning"]) + FEATURES, "afternoon": list(fd["features"]["afternoon"])}}
    p3 = run_variant(d, un, rules)
    p3.to_csv(OUT / "un_predictions.csv", index=False)
    a_un = pd.read_csv(OUTPUT_DIR / "experiments" / "E16" / "a_un_predictions.csv")
    a_un["date"] = pd.to_datetime(a_un["date"], utc=True).dt.tz_convert(tz)
    a_ud = s18.load("C1235", "E12_fd_lr", tz)
    e0 = _module("30_e16_up_notup.py").e0_window(d[KEYS + ["direction"]].assign(p_up=0), rules)
    y = p3[KEYS + ["direction", "p_up"]].rename(columns={"p_up": "p_E18"})
    for n, p in [("A-un", a_un), ("A-ud", a_ud), ("E0-un", e0)]:
        y = y.merge(p[KEYS + ["p_up"]].rename(columns={"p_up": f"p_{n}"}), on=KEYS)
    y = y[y["direction"].notna()].reset_index(drop=True)
    t3 = pd.concat([s31.compare(y, "E18", b, 0.95, n_boot) for b in ["A-un", "A-ud", "E0-un"]], ignore_index=True)
    t3.to_csv(RES / "h3_direction.csv", index=False, encoding="utf-8-sig")
    m3 = t3[(t3["b"] == "A-un") & (t3["group"] == "เช้า")].iloc[0]
    st3 = t3[(t3["b"] == "A-un") & (t3["group"].str.contains("KBANK|ADVANC"))]
    h3 = bool(m3["d_acc_lo"] > 0 and (st3["d_acc_hi"] >= 0).all() and m3["d_ll"] <= 0)
    pd.set_option("display.width", 220)
    print(pr[["model", "group", "n", "mae", "rmse", "share_moved", "moved_right", "d_mae", "d_mae_lo", "d_mae_hi", "d_rmse", "d_rmse_lo", "d_rmse_hi"]].round(4).to_string(index=False))
    print("H2", "ผ่าน" if h2 else "ไม่ผ่าน")
    print(t3[["a", "b", "group", "n", "acc_a", "acc_b", "pred_up_a", "d_ll", "d_ll_lo", "d_ll_hi", "d_acc", "d_acc_lo", "d_acc_hi"]].round(4).to_string(index=False))
    print("H3", "ผ่าน" if h3 else "ไม่ผ่าน")
    pd.DataFrame([{"H1": ok, "H2": h2, "H3": h3}]).to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
