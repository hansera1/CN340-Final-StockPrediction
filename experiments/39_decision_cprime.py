"""รอบ 7 ข้อ 9 ส่วนเพิ่ม: ตัวแทน C′ / C-adv / PR-C′ เทียบค่าตั้งต้น · CI 99.7% · ไม่สลับโมเดล

    python experiments/39_decision_cprime.py
→ results/decision_cprime.csv
"""
import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_rules
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


def main():
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    load = lambda n: (lambda p: p.assign(date=pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)))(pd.read_csv(OUTPUT_DIR / "experiments" / "E14" / n))
    pr, pp = load("cprime_predictions.csv"), load("prcprime_predictions.csv")
    a_un = pd.read_csv(OUTPUT_DIR / "experiments" / "E16" / "a_un_predictions.csv").assign(date=lambda p: pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz))
    a_ud, e0 = s18.load("C1235", "E12_fd_lr", tz), s30.e0_window(ds, rules)
    rep = pd.read_csv(ROOT / "results" / "e14" / "cprime_representatives.csv")
    rows = []
    for sess, th in [("morning", "เช้า"), ("afternoon", "บ่าย")]:
        cand = rep.loc[rep["ช่อง"] == f"{sess}_dir", "ตัวแทน"].iat[0]
        col = {"C′": "p_cprime", "C-adv": "p_cadv"}[cand]
        x = pr[pr["session"] == sess][KEYS + ["direction", col]].rename(columns={col: f"p_{cand}"})
        for n, p in [("A-ud", a_ud), ("A-un", a_un), ("E0-un", e0)]:
            x = x.merge(p[KEYS + ["p_up"]].rename(columns={"p_up": f"p_{n}"}), on=KEYS)
        x[f"p_{cand}"] = x[f"p_{cand}"].fillna(x["p_A-un"])
        x = x[x["direction"].notna()].reset_index(drop=True)
        c, e = s31.compare(x, cand, "A-ud", CONF, n_boot), s31.compare(x, cand, "E0-un", CONF, n_boot)
        a, st, ea = c[c["group"] == th].iloc[0], c[c["group"].str.contains("KBANK|ADVANC")], e[e["group"] == th].iloc[0]
        ok = bool(a["d_ll_hi"] < 0 and a["d_acc_hi"] >= 0 and (st["d_ll_lo"] <= 0).all() and (st["d_acc_hi"] >= 0).all())
        rows.append({"ช่อง": f"{sess}_dir", "ตัวแทน": cand, "แม่น": a["acc_a"], "แม่น_A-ud": a["acc_b"], "ทายขึ้น": a["pred_up_a"], "d_ll": a["d_ll"], "lo": a["d_ll_lo"], "hi": a["d_ll_hi"], "d_acc": a["d_acc"],
                     "ผ่าน": ok, "เทียบ_E0un_d_ll": ea["d_ll"], "E0un_lo": ea["d_ll_lo"], "E0un_hi": ea["d_ll_hi"]})
        eq, up = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr"]]
        y = ds[ds["session"] == sess][KEYS + ["p_ref", "p_end"]].merge(pp[KEYS + ["eq", "up"]], on=KEYS).merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(up[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS)
        y = y[y["p_end"].notna() & y["p_ref"].notna()].reset_index(drop=True)
        pref = y["p_ref"].to_numpy(float)
        y["tick1"] = False
        y["pt_PR-A"], y["pt_PR-C′"] = s18.price_rule(pref, y["eq_A"].to_numpy(), y["up_A"].to_numpy())[0], s18.price_rule(pref, y["eq"].to_numpy(), y["up"].to_numpy())[0]
        s32.CONF = CONF
        t = s32.score(y, ["PR-A", "PR-C′"], n_boot)
        g = t[(t["model"] == "PR-C′") & (t["ช่วง"] == "ทั้งช่วง")]
        a, st = g[g["group"] == th].iloc[0], g[g["group"].str.contains("KBANK|ADVANC")]
        ok = bool((a["d_mae_hi"] < 0 or a["d_rmse_hi"] < 0) and a["d_mae_lo"] <= 0 and a["d_rmse_lo"] <= 0 and (st["d_mae_lo"] <= 0).all() and (st["d_rmse_lo"] <= 0).all())
        rows.append({"ช่อง": f"{sess}_price", "ตัวแทน": "PR-C′", "MAE": a["mae"], "RMSE": a["rmse"], "d_mae": a["d_mae"], "mae_lo": a["d_mae_lo"], "mae_hi": a["d_mae_hi"], "d_rmse": a["d_rmse"], "rmse_lo": a["d_rmse_lo"], "rmse_hi": a["d_rmse_hi"], "ผ่าน": ok})
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results" / "decision_cprime.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(r.dropna(axis=1, how="all").round(4).T.to_string())


if __name__ == "__main__":
    main()
