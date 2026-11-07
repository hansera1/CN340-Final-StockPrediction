"""E25: ทดสอบซ้ำโมเดลราคาที่ขึ้นกับนิยามราคานิ่ง — เกณฑ์ใน results/e25_preregistration.md · รายงานเท่านั้น

    python experiments/52_e25_all_flat.py
→ outputs/experiments/E25/ (มีราคาจริง ไม่อยู่ใน git) · results/e25/ (ตัวเลขสรุปเท่านั้น)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecast.config import OUTPUT_DIR, ROOT, load_rules
from forecast.evaluation.splits import walk_forward_splits
from forecast.models.nested_tune import MODELS

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
TARGET_FROM = {"flat_dir": "flat_price", "updown": "price_updown"}
KS = (0.25, 0.5, 0.75, 1.0)
OUT, RES = OUTPUT_DIR / "experiments" / "E25", ROOT / "results" / "e25"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def add_family(x, family, flat, up, s18, expected_move):
    """flat, up: แถว KEYS + คอลัมน์ต่อแบบ → เพิ่ม pt_<family>:<แบบ>-rule และ -mean · ไม่มีผลแทนด้วย p_flat / p_ud"""
    names = [c for c in flat.columns if c not in KEYS and c in up.columns]
    a = x[KEYS].merge(flat[KEYS + names], on=KEYS, how="left")
    b = x[KEYS].merge(up[KEYS + names], on=KEYS, how="left")
    pc, r_up, r_dn = x["p_ref"].to_numpy(float), x["r_up"].to_numpy(float), x["r_dn"].to_numpy(float)
    fill, cols = [], {}
    for n in names:
        miss = a[n].isna() | b[n].isna()
        f, u = a[n].fillna(x["p_flat"]).to_numpy(float), b[n].fillna(x["p_ud"]).to_numpy(float)
        cols[f"pt_{family}:{n}-rule"] = s18.price_rule(pc, f, u)[0]
        cols[f"pt_{family}:{n}-mean"] = pc * np.exp(expected_move(f, u, r_up, r_dn))
        fill.append({"family": family, "model": n, "filled_share": float(miss.mean())})
    return pd.concat([x, pd.DataFrame(cols, index=x.index)], axis=1), fill


def verdict(t, names):
    q, rows = t.set_index(["model", "group"]), []
    for name in names:
        for _, th in SESSIONS:
            a = q.loc[(name, th)]
            k1 = bool(a["d_rmse_pra_hi"] < 0)
            k2 = all(q.loc[(name, f"{s} {th}"), "d_rmse_pra_lo"] <= 0 for s in ("KBANK", "ADVANC"))
            k3 = bool(a["d_mae_pra_lo"] <= 0)
            rows.append({"model": name, "session": th, "rmse": a["rmse"], "mae": a["mae"], "d_rmse": a["d_rmse_pra"], "lo": a["d_rmse_pra_lo"], "hi": a["d_rmse_pra_hi"], "d_mae": a["d_mae_pra"], "c1_rmse": k1, "c2_stocks": k2, "c3_mae": k3, "pass": k1 and k2 and k3})
    return pd.DataFrame(rows)


def ft_job(dp, cols, s, kind, target, te, s32, kappa30):
    if kappa30:
        import forecast.models.finetune_lr as f
        f.KAPPAS, f.FIXED_KAPPA = (30.0,), 30.0
    return s32.block(dp, cols, s, kind, target, te)


def run_ft(ds, exps, rules, s32):
    ev = rules["evaluation"]
    dp, cols = s32.prepared(ds, exps)
    jobs = []
    for s, _ in SESSIONS:
        days = dp.loc[dp["session"] == s, "date"].to_numpy()
        jobs += [(s, te) for _, te in walk_forward_splits(days, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
    parts = []
    # κ อัตโนมัติก่อน แล้วค่อย κ=30 เพราะการตั้งค่า κ ติดอยู่ใน worker
    for tag, kind, k30 in [("D", "D", False), ("D+C", "D+C", False), ("D-k30", "D", True)]:
        res = Parallel(n_jobs=-1)(delayed(ft_job)(dp, cols[TARGET_FROM[t]], s, kind, t, te, s32, k30) for s, te in jobs for t in TARGET_FROM)
        parts.append(pd.concat(res, ignore_index=True).assign(tag=tag))
    p = pd.concat(parts, ignore_index=True)
    wide = {}
    for t in TARGET_FROM:
        wide[t] = p[p["target"] == t].pivot_table(index=KEYS, columns="tag", values="p").reset_index()
        wide[t].columns.name = None
    return wide


def run_tab(d, exps, rules, s33, s42, s43, params=None):
    if params:
        s43.PARAMS, s43.K_TOP, s43.K_C = params
    starts = s42.starts_by_session()
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")["features"]
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    daily = [c for c in fd["morning"] if c.startswith(("fd_", "fl_dow"))]
    eq, up = list(v["aprice_lr"]["features"]), v["pprice_lr"]["features"]
    sets = {"flat_dir": {s: list(dict.fromkeys(eq + daily)) + ["d_sym_advanc"] for s, _ in SESSIONS},
            "updown": {s: list(dict.fromkeys(list(up[s]) + daily)) + ["d_sym_advanc"] for s, _ in SESSIONS}}
    return {t: s43.run_tab(d, c, t, s43.ALL, rules, starts)[0].drop(columns="E0") for t, c in sets.items()}


def run_e17b(d, exps, rules, s33):
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")["features"]["morning"]
    daily = [c for c in fd if c.startswith(("fd_", "fl_dow"))]
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    eq, up = list(v["aprice_lr"]["features"]), v["pprice_lr"]["features"]
    cols = {"flat_dir": {s: list(dict.fromkeys(eq + daily)) + ["d_sym_advanc"] for s, _ in SESSIONS},
            "updown": {s: list(dict.fromkeys(list(up[s]) + daily)) + ["d_sym_advanc"] for s, _ in SESSIONS}}
    out, tuning = {}, []
    for t, c in cols.items():
        p, info, _ = s33.run(d, c, rules, t)
        a = pd.concat([p, s33.ensembles(p, info)], ignore_index=True)
        tuning.append(info.assign(params=info["params"].astype(str), features=info["features"].astype(str)))
        w = a[a["split"] == "wf"].pivot_table(index=KEYS, columns="model", values="p").reset_index()
        w.columns.name = None
        out[t] = w
    return out, pd.concat(tuning, ignore_index=True)


def main():
    t0 = time.perf_counter()
    s48, s47, s18, s32, s33, s42, s24 = [_module(n) for n in ["48_e23_skip_fx.py", "47_e22_shrink_k.py", "18_score_round4.py", "32_e17a_price.py", "33_e15_models.py", "42_overfit_audit.py", "51_e24_two_step.py"]]
    s43 = _module("43_e20_antioverfit.py")
    rules = load_rules()
    tz, ev, n_boot = rules["timezone"], rules["evaluation"], rules["evaluation"]["n_boot"]
    ds, exps, s31, _, s38, _ = s48.prepare(rules)
    d, _ = s31.aligned(ds, exps)
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(OUTPUT_DIR / "experiments" / "E24" / "points.csv")
    x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_convert(tz)
    assert len(x) == 2188
    base = ["pt_PR-A", "pt_P(c)"]
    x = x[KEYS + ["p_ref", "p_end", "p_flat", "p_ud", "r_up", "r_dn"] + base].copy()

    fills = []
    e17b, tuning = run_e17b(d, exps, rules, s33)
    tuning.to_csv(OUT / "e17b_tuning.csv", index=False)
    x, f = add_family(x, "E17b", e17b["flat_dir"], e17b["updown"], s18, s24.expected_move)
    fills += f
    print(f"E17b เสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)
    tab = run_tab(d, exps, rules, s33, s42, s43)
    x, f = add_family(x, "E20", tab["flat_dir"], tab["updown"], s18, s24.expected_move)
    fills += f
    print(f"E20 เสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)
    tabb = run_tab(d, exps, rules, s33, s42, s43, (s43.PARAMS2, 5, 0.001))
    x, f = add_family(x, "E20b", tabb["flat_dir"], tabb["updown"], s18, s24.expected_move)
    fills += f
    print(f"E20b เสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)
    ft = run_ft(ds, exps, rules, s32)
    x, f = add_family(x, "FT", ft["flat_dir"], ft["updown"], s18, s24.expected_move)
    fills += f
    print(f"fine-tune เสร็จ {time.perf_counter() - t0:.0f} วิ", flush=True)

    x["ez"] = s24.expected_move(x["p_flat"].to_numpy(float), x["p_ud"].to_numpy(float), x["r_up"].to_numpy(float), x["r_dn"].to_numpy(float))
    pc = x["p_ref"].to_numpy(float)
    for k in KS:
        x[f"pt_E22:k={k:g}"] = s47.shrink_point(pc, x["ez"].to_numpy(float), k)
    x["k"] = np.nan
    for sess, _ in SESSIONS:
        m = (x["session"] == sess).to_numpy()
        g = x[m].reset_index(drop=True)
        days = d.loc[d["session"] == sess, "date"].to_numpy()
        blocks = [te for _, te in walk_forward_splits(days, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"])]
        x.loc[m, "k"] = s47.walk_forward_k(g, blocks).to_numpy()
    assert x["k"].notna().all()
    x["pt_E22:PR-k"] = s47.shrink_point(pc, x["ez"].to_numpy(float), x["k"].to_numpy(float))
    x = x.copy()
    x.to_csv(OUT / "points.csv", index=False)

    names = [c[3:] for c in x.columns if c.startswith("pt_") and c not in base]
    t = s47.score(x, ["PR-A", "P(c)"] + names, n_boot, s38.boot_rmse, s31.boot_ci)
    got = t[t["group"].isin(["เช้า", "บ่าย"]) & t["model"].isin(["PR-A", "P(c)"])].set_index(["model", "group"])["rmse"].round(3).to_dict()
    assert got[("PR-A", "เช้า")] == 3.083 and got[("PR-A", "บ่าย")] == 1.830 and got[("P(c)", "เช้า")] == 3.073 and got[("P(c)", "บ่าย")] == 1.817, got
    v = verdict(t, names)
    t.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    v.to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")
    fill = pd.DataFrame(fills)
    fill.to_csv(RES / "filled.csv", index=False, encoding="utf-8-sig")
    x.groupby(["session", "k"]).agg(n=("k", "size"), n_days=("date", "nunique")).reset_index().to_csv(RES / "k_selected.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print("จำนวนแบบ", len(names), "· ผ่าน", int(v["pass"].sum()), "จาก", len(v))
    print(v[v["pass"]].round(4).to_string(index=False))
    for _, th in SESSIONS:
        print("--- ดีสุด", th)
        print(v[v["session"] == th].sort_values("rmse").head(12).round(4).to_string(index=False))
    print(v.groupby("session")[["rmse", "mae"]].describe().T.round(3).to_string())
    print("สัดส่วนแถวที่แทนค่า", fill[fill["filled_share"] > 0].round(3).to_string(index=False))
    print(f"รวม {time.perf_counter() - t0:.0f} วิ", flush=True)


if __name__ == "__main__":
    main()
