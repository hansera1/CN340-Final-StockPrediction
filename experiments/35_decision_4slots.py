"""รอบ 7 ข้อ 9: เทียบตัวแทนกับค่าตั้งต้นทีละช่อง — กติกาใน results/decision_rules.md · ไม่สลับโมเดลเอง

CI 99.7% (≤ 16 การเทียบ) · ทิศทางใช้เกณฑ์ทิศทาง · ราคาต้องไม่แย่ลงทั้ง MAE และ RMSE · ต้องไม่แย่กว่าในหุ้นใดหุ้นหนึ่ง

    python experiments/35_decision_4slots.py
→ results/decision_4slots.csv
"""
import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset

warnings.filterwarnings("ignore")


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s17, s18, s30, s31, s32 = [_module(n) for n in ["17_label_variants.py", "18_score_round4.py", "30_e16_up_notup.py", "31_e14_finetune.py", "32_e17a_price.py"]]
KEYS = ["date", "symbol", "session"]
CONF = 0.997
EXP = OUTPUT_DIR / "experiments"
RES = ROOT / "results"
SESS = {"morning": "เช้า", "afternoon": "บ่าย"}


def rep_of(path, slot):
    t = pd.read_csv(path)
    return t.loc[t["ช่อง"] == slot, "ตัวแทน"].iat[0]


def loaded(path, tz):
    p = pd.read_csv(path)
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def main():
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    d, _ = s31.aligned(ds, load_experiments())
    rows = []

    # ---------- ช่อง 1–2 ทิศทาง
    e14 = {"D": loaded(EXP / "E14" / "d_predictions.csv", tz), "D+C": loaded(EXP / "E14" / "dc_predictions.csv", tz)}
    e15 = loaded(EXP / "E15" / "predictions_all.csv", tz)
    e15 = e15[e15["split"] == "wf"]
    a_ud = s18.load("C1235", "E12_fd_lr", tz)
    e0 = s30.e0_window(ds, rules)
    for sess, th in SESS.items():
        slot = f"{sess}_dir"
        r14, r15 = rep_of(RES / "e14" / "representatives.csv", slot), rep_of(RES / "e15" / "representatives.csv", slot)
        x = d[d["session"] == sess][KEYS + ["direction"]]
        for n, p in [("A-ud", a_ud), ("E0-un", e0), (f"E14 {r14}", e14[r14]), (f"E15 {r15}", e15[e15["model"] == r15].rename(columns={"p": "p_up"}))]:
            x = x.merge(p[KEYS + ["p_up"]].rename(columns={"p_up": f"p_{n}"}), on=KEYS)
        x = x[x["direction"].notna()].reset_index(drop=True)
        x[f"p_E15 {r15}"] = x[f"p_E15 {r15}"].fillna(x["p_E0-un"])
        for cand in [f"E14 {r14}", f"E15 {r15}"]:
            c = s31.compare(x, cand, "A-ud", CONF, n_boot)
            e = s31.compare(x, cand, "E0-un", CONF, n_boot)
            all_, stocks = c[c["group"] == th].iloc[0], c[c["group"].str.contains("KBANK|ADVANC")]
            e_all = e[e["group"] == th].iloc[0]
            win = all_["d_ll_hi"] < 0 and all_["d_acc_hi"] >= 0
            stock_ok = bool((stocks["d_ll_lo"] <= 0).all() and (stocks["d_acc_hi"] >= 0).all())
            rows.append({"ช่อง": slot, "ตัวแทน": cand, "ค่าตั้งต้น": "A-ud", "แม่น_ตัวแทน": all_["acc_a"], "แม่น_ค่าตั้งต้น": all_["acc_b"], "ทาย_ขึ้น": all_["pred_up_a"],
                         "d_logloss": all_["d_ll"], "lo": all_["d_ll_lo"], "hi": all_["d_ll_hi"], "d_acc": all_["d_acc"], "acc_lo": all_["d_acc_lo"], "acc_hi": all_["d_acc_hi"],
                         "ไม่แย่กว่ารายหุ้น": stock_ok, "ผ่าน": bool(win and stock_ok),
                         "เทียบ_E0un_d_ll": e_all["d_ll"], "E0un_lo": e_all["d_ll_lo"], "E0un_hi": e_all["d_ll_hi"], "เทียบ_E0un_d_acc": e_all["d_acc"]})

    # ---------- ช่อง 3–4 ราคา
    s32.CONF = CONF
    a = pd.read_csv(EXP / "E17a" / "ft_predictions.csv")
    a["date"] = pd.to_datetime(a["date"], utc=True).dt.tz_convert(tz)
    b = {t: loaded(EXP / "E17b" / f"{t}_predictions.csv", tz) for t in ["flat_price", "price_updown"]}
    b = {t: v[v["split"] == "wf"] for t, v in b.items()}
    eq, pp = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr"]]
    h = d[KEYS + ["z"]].sort_values(["symbol", "session", "date"]).copy()
    pos, neg = h["z"].where(h["z"] > 0), h["z"].where(h["z"] < 0)
    h["z_up"] = pos.groupby([h["symbol"], h["session"]]).transform(lambda s: s.expanding().mean().shift(1))
    h["z_dn"] = neg.groupby([h["symbol"], h["session"]]).transform(lambda s: s.expanding().mean().shift(1))
    for sess, th in SESS.items():
        slot = f"{sess}_price"
        ra, rb = rep_of(RES / "e17a" / "representatives.csv", slot), rep_of(RES / "e17b" / "representatives.csv", slot)
        kind = ra.replace("PR-", "")
        x = d[d["session"] == sess][KEYS + ["p_ref", "p_end", "tick1"]].merge(h[KEYS + ["z_up", "z_dn"]], on=KEYS)
        x = x.merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS)
        for c, t in [("eq", "flat_price"), ("up", "price_updown")]:
            x = x.merge(a[(a["kind"] == kind) & (a["target"] == t)][KEYS + ["p"]].rename(columns={"p": f"{c}_a"}), on=KEYS)
            x = x.merge(b[t][b[t]["model"] == rb][KEYS + ["p"]].rename(columns={"p": f"{c}_b"}), on=KEYS)
        x = x[x["p_end"].notna() & x["p_ref"].notna() & x["z_up"].notna()].reset_index(drop=True)
        pref = x["p_ref"].to_numpy(float)
        x["pt_PR-A"] = s18.price_rule(pref, x["eq_A"].to_numpy(), x["up_A"].to_numpy())[0]
        x["pt_P(c)"] = pref
        x[f"pt_E17a {ra}"] = s18.price_rule(pref, x["eq_a"].to_numpy(), x["up_a"].to_numpy())[0]
        x[f"pt_E17b {rb}"] = s18.price_rule(pref, x["eq_b"].fillna(x["eq_A"]).to_numpy(), x["up_b"].fillna(x["up_A"]).to_numpy())[0]
        x["pt_PR-mean"] = pref * np.exp((1 - x["eq_A"]) * x["up_A"] * x["z_up"] + (1 - x["eq_A"]) * (1 - x["up_A"]) * x["z_dn"])
        names = ["PR-A", "P(c)", f"E17a {ra}", f"E17b {rb}", "PR-mean"]
        t = s32.score(x, names, n_boot)
        t.to_csv(RES / f"decision_{slot}.csv", index=False, encoding="utf-8-sig")
        base = t[(t["model"] == "PR-A") & (t["ช่วง"] == "ทั้งช่วง") & (t["group"] == th)].iloc[0]
        for cand in names[2:]:
            g = t[(t["model"] == cand) & (t["ช่วง"] == "ทั้งช่วง")]
            all_, stocks = g[g["group"] == th].iloc[0], g[g["group"].str.contains("KBANK|ADVANC")]
            win = all_["d_mae_hi"] < 0 or all_["d_rmse_hi"] < 0
            not_worse = all_["d_mae_lo"] <= 0 and all_["d_rmse_lo"] <= 0
            stock_ok = bool((stocks["d_mae_lo"] <= 0).all() and (stocks["d_rmse_lo"] <= 0).all())
            rows.append({"ช่อง": slot, "ตัวแทน": cand, "ค่าตั้งต้น": "PR-A", "MAE_ตัวแทน": all_["mae"], "MAE_ค่าตั้งต้น": base["mae"], "RMSE_ตัวแทน": all_["rmse"], "RMSE_ค่าตั้งต้น": base["rmse"],
                         "d_mae": all_["d_mae"], "mae_lo": all_["d_mae_lo"], "mae_hi": all_["d_mae_hi"], "d_rmse": all_["d_rmse"], "rmse_lo": all_["d_rmse_lo"], "rmse_hi": all_["d_rmse_hi"],
                         "ไม่แย่กว่ารายหุ้น": stock_ok, "ผ่าน": bool(win and not_worse and stock_ok)})
    r = pd.DataFrame(rows)
    r.to_csv(RES / "decision_4slots.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(r[r["ช่อง"].str.endswith("_dir")].dropna(axis=1, how="all").round(4).T.to_string())
    print(r[r["ช่อง"].str.endswith("_price")].dropna(axis=1, how="all").round(4).T.to_string())


if __name__ == "__main__":
    main()
