"""E19 รันสำรวจ: เทียบภาพ 5m กับ 1m แถวต่อแถว — ไม่ใช้ตัดสิน

    python experiments/40_e19_compare.py
ต้องรันก่อน: 28_e19_train.py --explore 8 และ --explore 8 --minutes 5
→ results/e19/compare_5m_vs_1m.csv
"""
import importlib

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT

day_ci = importlib.import_module("28_e19_train").day_ci
KEY = ["arch", "kind", "date", "symbol"]


def load(name):
    r = pd.read_csv(OUTPUT_DIR / "experiments" / name / "predictions.csv")
    y = (r["direction"] == 1).astype(float)
    p = r["p_up"].clip(1e-6, 1 - 1e-6)
    return r.assign(ll=-(y * np.log(p) + (1 - y) * np.log(1 - p)), ok=((r["p_up"] > .5) == (y == 1)).astype(float), ae=(r["p_end"] - r["price"]).abs(), up=(r["p_up"] > .5).astype(float))


def compare(a, b):
    """a = 1m, b = 5m · ต้องเป็นแถวและป้ายชุดเดียวกัน"""
    j = a.merge(b, on=KEY, suffixes=("_1m", "_5m"), validate="one_to_one")
    assert len(j) == len(a) == len(b) and (j["direction_1m"] == j["direction_5m"]).all() and (j["p_end_1m"] == j["p_end_5m"]).all()
    rows = []
    for (arch, kind), g in j.groupby(["arch", "kind"]):
        row = {"arch": arch, "kind": kind, "n": len(g), "days": g["date"].nunique(), "pred_up_1m": g["up_1m"].sum(), "pred_up_5m": g["up_5m"].sum()}
        for m in ("ll", "ok", "ae"):
            row[f"{m}_1m"], row[f"{m}_5m"] = g[f"{m}_1m"].mean(), g[f"{m}_5m"].mean()
            row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = day_ci((g[f"{m}_5m"] - g[f"{m}_1m"]).to_numpy(), g["date"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    t = compare(load("E19_explore"), load("E19_explore_5m"))
    t.to_csv(ROOT / "results" / "e19" / "compare_5m_vs_1m.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    cols = ["arch", "kind", "n", "pred_up_1m", "pred_up_5m", "ok_1m", "ok_5m", "ll_1m", "ll_5m", "d_ll", "d_ll_lo", "d_ll_hi", "ae_1m", "ae_5m", "d_ae", "d_ae_lo", "d_ae_hi"]
    print(t[cols].round(3).to_string(index=False))
    print("** สำรวจ: 12 แถว 6 วัน ข้อมูลน้อยเกินสรุป ไม่ใช้ตัดสิน **")


if __name__ == "__main__":
    main()
