"""E17b (รอบ 7 ข้อ 8): วิธีของ E15 กับ A-price และ p_price — เกณฑ์ใน results/e17b_preregistration.md

    python experiments/34_e17b_price_models.py
→ outputs/experiments/E17b/*.csv, results/e17b/*.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset
from forecast.models.nested_tune import MODELS

warnings.filterwarnings("ignore")


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


KEYS = ["date", "symbol", "session"]
QC = ["q10", "q25", "q50", "q75", "q90"]


def main():
    s17, s18, s31, s32, s33 = [_module(n) for n in ["17_label_variants.py", "18_score_round4.py", "31_e14_finetune.py", "32_e17a_price.py", "33_e15_models.py"]]
    s32.CONF = 1 - 0.05 / 22
    rules = load_rules()
    tz, ev = rules["timezone"], rules["evaluation"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    exps = load_experiments()
    d, _ = s31.aligned(ds, exps)
    v = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    daily = list(next(x for x in exps["E12"]["variants"] if x["name"] == "fd_lr")["features"]["morning"])
    daily = [c for c in daily if c.startswith(("fd_", "fl_dow"))]
    eq_base, up_base = list(v["aprice_lr"]["features"]), v["pprice_lr"]["features"]
    cols = {"flat_price": {s: list(dict.fromkeys(eq_base + daily)) + ["d_sym_advanc"] for s in ("morning", "afternoon")},
            "price_updown": {s: list(dict.fromkeys(list(up_base[s]) + daily)) + ["d_sym_advanc"] for s in ("morning", "afternoon")}}
    OUT, RES = OUTPUT_DIR / "experiments" / "E17b", ROOT / "results" / "e17b"
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    pred, inner = {}, {}
    for target, c in cols.items():
        p, info, oof = s33.run(d, c, rules, target)
        print(target, "แถว", len(p), "trial", int(info["n_trials"].sum()), flush=True)
        a = pd.concat([p, s33.ensembles(p, info)], ignore_index=True)
        a.to_csv(OUT / f"{target}_predictions.csv", index=False)
        info.assign(params=info["params"].astype(str), features=info["features"].astype(str)).to_csv(OUT / f"{target}_tuning.csv", index=False)
        pred[target] = a[a["split"] == "wf"].pivot_table(index=KEYS, columns="model", values="p").reset_index()
        inner[target] = {s: s33.inner_scores(oof, info, s).set_index("period") for s in ("morning", "afternoon")}
    names = MODELS + ["ENS-2", "ENS-all", "ENS-w", "ENS-top"]

    eq, pp, zq = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr", "E9_baseline_ud"]]
    x = d[KEYS + ["p_ref", "p_end", "z", "tick1"]].merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS)
    x = x.merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS).merge(zq[KEYS + QC], on=KEYS)
    x = x.merge(pred["flat_price"].rename(columns={n: f"eq_{n}" for n in names}), on=KEYS).merge(pred["price_updown"].rename(columns={n: f"up_{n}" for n in names}), on=KEYS)
    h = d[KEYS + ["z"]].sort_values(["symbol", "session", "date"]).copy()
    pos, neg = h["z"].where(h["z"] > 0), h["z"].where(h["z"] < 0)
    h["z_up"] = pos.groupby([h["symbol"], h["session"]]).transform(lambda s: s.expanding().mean().shift(1))
    h["z_dn"] = neg.groupby([h["symbol"], h["session"]]).transform(lambda s: s.expanding().mean().shift(1))
    x = x.merge(h[KEYS + ["z_up", "z_dn"]], on=KEYS)
    x = x[x["p_end"].notna() & x["p_ref"].notna() & x["z_up"].notna()].reset_index(drop=True)
    pref = x["p_ref"].to_numpy(float)
    x["pt_PR-A"] = s18.price_rule(pref, x["eq_A"].to_numpy(), x["up_A"].to_numpy())[0]
    x["pt_P(c)"] = pref
    scored = ["PR-A", "P(c)"]
    for n in names:
        a, b = x[f"eq_{n}"].fillna(x["eq_A"]).to_numpy(), x[f"up_{n}"].fillna(x["up_A"]).to_numpy()
        x[f"pt_{n}"] = s18.price_rule(pref, a, b)[0]
        x[f"pt_{n}-mean"] = pref * np.exp((1 - a) * b * x["z_up"] + (1 - a) * (1 - b) * x["z_dn"])
        scored += [n, f"{n}-mean"]
    print("เทียบบน", len(x), "รอบ", flush=True)
    t = s32.score(x, scored, ev["n_boot"])
    t.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    cov = s32.coverage(x, ["PR-A"] + names, x[QC], "ทุกแถว")
    cov.to_csv(RES / "coverage.csv", index=False, encoding="utf-8-sig")

    get = lambda n, span, g: t[(t["model"] == n) & (t["ช่วง"] == span) & (t["group"] == g)].iloc[0]
    gap = lambda n, g, c, tgt: abs(cov[(cov["model"] == n) & (cov["group"] == g)][c].iat[0] - tgt)
    rep, ver = [], []
    for sess, g in [("morning", "เช้า"), ("afternoon", "บ่าย")]:
        s = inner["flat_price"][sess].add(inner["price_updown"][sess]).dropna()
        cand = [c for c in s.columns if c in names]
        mean = s[cand].mean()
        rep.append({"ช่อง": f"{sess}_price", "ตัวแทน": mean.idxmin(), "inner": mean.min(), **{f"inner_{c}": mean[c] for c in cand}})
        for n in names:
            cn = n
            for nm, coverage_name in [(n, cn), (f"{n}-mean", None)]:
                w, o = get(nm, "ทั้งช่วง", g), get(nm, "ช่องราคา 1 บาท", g)
                win = (w["d_mae_hi"] < 0 and w["d_rmse_lo"] <= 0) or (w["d_rmse_hi"] < 0 and w["d_mae_lo"] <= 0)
                other = o["d_mae_lo"] <= 0 and o["d_rmse_lo"] <= 0
                cover = True if coverage_name is None else (gap(n, g, "cover50", .5) <= gap("PR-A", g, "cover50", .5) and gap(n, g, "cover80", .8) <= gap("PR-A", g, "cover80", .8))
                ver.append({"รอบ": g, "แบบ": nm, "d_mae": w["d_mae"], "mae_lo": w["d_mae_lo"], "mae_hi": w["d_mae_hi"], "d_rmse": w["d_rmse"], "rmse_lo": w["d_rmse_lo"],
                            "rmse_hi": w["d_rmse_hi"], "ชนะ": bool(win), "อีกช่วงไม่แย่": bool(other), "coverage": bool(cover), "ผ่าน": bool(win and other and cover)})
    rep, ver = pd.DataFrame(rep), pd.DataFrame(ver)
    rep.to_csv(RES / "representatives.csv", index=False, encoding="utf-8-sig")
    ver.to_csv(RES / "verdict.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(ver[ver["แบบ"].isin(["lr", "enet", "lgbm", "xgb", "rf", "svm", "mlp", "ENS-all", "ENS-w", "ENS-top", "mlp-mean", "ENS-w-mean"])].round(4).to_string(index=False))
    print("\nตัวแทน\n", rep.round(4).T.to_string())
    print("ผ่านทั้งหมด", int(ver["ผ่าน"].sum()), "จาก", len(ver))


if __name__ == "__main__":
    main()
