"""E24: โมเดลราคา 2 ขั้น ใช้นิยามราคานิ่งเทียบราคาเปิด — เกณฑ์ใน results/e24_preregistration.md · รายงานเท่านั้น

    python experiments/51_e24_two_step.py
→ outputs/experiments/E24/ (มีราคาจริง ไม่อยู่ใน git) · results/e24/ (ตัวเลขสรุปเท่านั้น)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_rules
from forecast.evaluation.metrics import round_to_tick
from forecast.evaluation.runner import run_variant

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
MODELS = ["S0-rule", "S0-mean", "Sg-rule", "Sg-mean"]
REPRESENTATIVE = {"morning": "Sg-mean", "afternoon": "S0-mean"}
OUT, RES = OUTPUT_DIR / "experiments" / "E24", ROOT / "results" / "e24"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cond_r(d):
    """E[r|ขึ้น], E[r|ลง] จากวันก่อนวันทาย แยกหุ้นและรอบ · r = ln(ราคาปลาย / ราคาเปิด)"""
    h = d[KEYS + ["p_start", "p_end"]].sort_values(["symbol", "session", "date"]).copy()
    r = np.log(h["p_end"].astype(float) / h["p_start"].astype(float))
    grp = [h["symbol"], h["session"]]
    h["r_up"] = r.where(r > 0).groupby(grp).transform(lambda s: s.expanding().mean().shift(1))
    h["r_dn"] = r.where(r < 0).groupby(grp).transform(lambda s: s.expanding().mean().shift(1))
    return h[KEYS + ["r_up", "r_dn"]]


def expected_move(p_flat, p_up, r_up, r_dn):
    return (1 - p_flat) * (p_up * r_up + (1 - p_up) * r_dn)


def add_points(x, s18):
    """ต้องมี p_ref, g_hat, p_flat, p_ud, r_up, r_dn, session · แบบ Sg เฉพาะรอบเช้า"""
    pc, g = x["p_ref"].to_numpy(float), x["g_hat"].to_numpy(float)
    flat, ud = x["p_flat"].to_numpy(float), x["p_ud"].to_numpy(float)
    move = expected_move(flat, ud, x["r_up"].to_numpy(float), x["r_dn"].to_numpy(float))
    am = (x["session"] == "morning").to_numpy()
    base = np.asarray(round_to_tick(pc * np.exp(g)), float)
    x["pt_S0-rule"] = s18.price_rule(pc, flat, ud)[0]
    x["pt_S0-mean"] = pc * np.exp(move)
    x["pt_Sg-rule"] = np.where(am, s18.price_rule(base, flat, ud)[0], np.nan)
    x["pt_Sg-mean"] = np.where(am, pc * np.exp(g + move), np.nan)
    x["start_guess_Sg"] = np.where(am, pc * np.exp(g), np.nan)
    return x


def verdict(t):
    q, rows = t.set_index(["model", "group"]), []
    for sess, th in SESSIONS:
        m = REPRESENTATIVE[sess]
        a = q.loc[(m, th)]
        k1 = bool(a["d_rmse_pra_hi"] < 0)
        k2 = all(q.loc[(m, f"{s} {th}"), "d_rmse_pra_lo"] <= 0 for s in ("KBANK", "ADVANC"))
        k3 = bool(a["d_mae_pra_lo"] <= 0)
        rows.append({"session": th, "model": m, "c1_rmse": k1, "c2_stocks": k2, "c3_mae": k3, "pass": k1 and k2 and k3})
    return pd.DataFrame(rows)


def score_all(x, n_boot, s47, s38, s31):
    out = [s47.score(x, ["PR-A", "P(c)"], n_boot, s38.boot_rmse, s31.boot_ci)]
    for name in MODELS:
        ok = x[f"pt_{name}"].notna().to_numpy()
        out.append(s47.score(x[ok].reset_index(drop=True), [name], n_boot, s38.boot_rmse, s31.boot_ci))
    return pd.concat(out, ignore_index=True)


def decomposition(x):
    """ส่วนราคาเปิด = P(เปิด) − ราคาจุด · ส่วนหลังเปิด = ราคาปลาย − P(เปิด)"""
    rows = []
    h = x[x["p_start"].notna()]
    for sess, th in SESSIONS:
        s = h[h["session"] == sess]
        for name in ["PR-A", REPRESENTATIVE[sess]]:
            pt = s[f"pt_{name}"].to_numpy(float)
            a, b = s["p_start"].to_numpy(float) - pt, (s["p_end"] - s["p_start"]).to_numpy(float)
            rows.append({"session": th, "model": name, "n": len(s), "rmse_start_part": np.sqrt((a ** 2).mean()), "rmse_after_open": np.sqrt((b ** 2).mean()), "rmse_total": np.sqrt(((a + b) ** 2).mean())})
    return pd.DataFrame(rows)


def gap_for(data, s38, s37):
    from forecast.features.overnight import build_features
    tables, hourly = s37.load_tables()
    fx = build_features(pd.DatetimeIndex(np.sort(data["date"].unique())), tables, hourly)
    am, _ = s38.morning_frame(data, fx)
    gh, _ = s38.gap_oos(am)
    s38.assert_no_leak(gh)
    return gh[KEYS + ["g_hat"]]


def window_rows(data, days, flat_var, ud_var, exps, rules, s47, s18, s38, s37):
    """holdout / ข้อมูลใหม่: โมเดลฝึกก่อนช่วงนี้ · รายงานเท่านั้น"""
    r = copy.deepcopy(rules)
    r["evaluation"]["holdout_days"] = days
    w = s47.report_window(data, days, exps, rules, {"morning": 0.0, "afternoon": 0.0}, s18)
    flat = run_variant(data, flat_var, r, holdout_eval=True)[KEYS + ["p_up"]].rename(columns={"p_up": "p_flat"})
    ud = run_variant(data, ud_var, r, holdout_eval=True)[KEYS + ["p_up"]].rename(columns={"p_up": "p_ud"})
    w = w.merge(flat, on=KEYS).merge(ud, on=KEYS).merge(cond_r(data), on=KEYS).merge(data[KEYS + ["p_start"]], on=KEYS).merge(gap_for(data, s38, s37), on=KEYS, how="left")
    w["g_hat"] = w["g_hat"].fillna(0.0)
    return add_points(w, s18)


def main():
    s48, s47, s18, s37 = [_module(n) for n in ["48_e23_skip_fx.py", "47_e22_shrink_k.py", "18_score_round4.py", "37_e18_overnight.py"]]
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    ds, exps, s31, _, s38, _ = s48.prepare(rules)
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    flat_var = v["adir_lr"]
    fd = next(x for x in exps["E12"]["variants"] if x["name"] == "fd_lr")
    ud_var = {**fd, "params": {**fd.get("params", {}), "permutation": False}}

    d, _ = s31.aligned(ds, exps)
    eq, pp, zq, ud = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr", "E9_baseline_ud", "E12_fd_lr"]]
    x = d[KEYS + ["p_ref", "p_end", "p_start"]].merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS).merge(zq[KEYS], on=KEYS)
    x = x.merge(s47.cond_means(d), on=KEYS)
    x = x[x["p_end"].notna() & x["p_ref"].notna() & x["z_up"].notna() & x["z_dn"].notna()].sort_values(KEYS).reset_index(drop=True)
    assert len(x) == 2188, len(x)
    x = s47.attach_points(x, s18)
    flat = run_variant(ds, flat_var, rules)[KEYS + ["p_up"]].rename(columns={"p_up": "p_flat"})
    flat.to_csv(OUT / "flat_predictions.csv", index=False)
    print("P(นิ่ง) ฝึกเสร็จ", len(flat), flush=True)
    x = x.merge(flat, on=KEYS).merge(ud[KEYS + ["p_up"]].rename(columns={"p_up": "p_ud"}), on=KEYS).merge(cond_r(d), on=KEYS).merge(gap_for(ds, s38, s37), on=KEYS, how="left")
    assert len(x) == 2188, ("เหลือแถวไม่ครบ", len(x))
    x["g_hat"] = x["g_hat"].fillna(0.0)
    x = add_points(x, s18)
    x.to_csv(OUT / "points.csv", index=False)

    t = score_all(x, n_boot, s47, s38, s31)
    got = t[t["group"].isin(["เช้า", "บ่าย"]) & t["model"].isin(["PR-A", "P(c)"])].set_index(["model", "group"])["rmse"].round(3).to_dict()
    assert got[("PR-A", "เช้า")] == 3.083 and got[("PR-A", "บ่าย")] == 1.830 and got[("P(c)", "เช้า")] == 3.073 and got[("P(c)", "บ่าย")] == 1.817, got
    verd, dec = verdict(t), decomposition(x)
    for name, tab in [("metrics", t), ("verdict", verd), ("decomposition", dec)]:
        tab.to_csv(RES / f"{name}.csv", index=False, encoding="utf-8-sig")

    new_ds = s48._load_new(rules)
    windows = {"holdout": window_rows(ds, 10, flat_var, ud_var, exps, rules, s47, s18, s38, s37), "new_data": window_rows(new_ds, 6, flat_var, ud_var, exps, rules, s47, s18, s38, s37)}
    windows["new_data"].to_csv(OUT / "new_rows.csv", index=False)
    tabs = {k: score_all(w, n_boot, s47, s38, s31) for k, w in windows.items()}
    for k, tab in tabs.items():
        tab.to_csv(RES / f"{k}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    show = ["model", "group", "n", "rmse", "mae", "d_rmse_pra", "d_rmse_pra_lo", "d_rmse_pra_hi", "d95_rmse_pra_lo", "d95_rmse_pra_hi", "d_mae_pra", "d_mae_pra_lo", "d_mae_pra_hi", "d_rmse_pc", "d_rmse_pc_lo", "d_rmse_pc_hi"]
    print(t[show].round(3).to_string(index=False))
    print(verd.to_string(index=False))
    print(dec.round(3).to_string(index=False))
    for k, tab in tabs.items():
        print("---", k)
        print(tab[tab["group"].isin(["เช้า", "บ่าย"])][["model", "group", "n", "rmse", "mae", "d_rmse_pra", "d_rmse_pra_lo", "d_rmse_pra_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
