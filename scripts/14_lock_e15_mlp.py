"""รอบ 7 ข้อ 9: ล็อก E15 MLP ช่องทิศทางรอบบ่าย ตามที่เจ้าของโปรเจกต์สั่ง — ขั้นตอนเดียวกับ walk-forward ใน experiments/33_e15_models.py

จูนด้วยข้อมูลที่มีทั้งหมด (inner CV 4 fold, Optuna ≤ 25 trial, เลือก k) แล้ว fit บนรอบบ่ายทุกรอบที่รู้ผล
ต้องรันใน environment ที่ตรง requirements-live.txt

    python scripts/14_lock_e15_mlp.py
→ outputs/models/R4ATC_UN/E15/mlp/<SYMBOL>_afternoon.pkl (bundle kind = "sklearn")
"""
import pickle
from datetime import datetime

import _setup  # noqa: F401
import numpy as np
import sklearn

from forecast.config import OUTPUT_DIR, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset
from forecast.models.e9 import target_rows
from forecast.models.nested_tune import build, proba, rank_features, tune

SYM_COL = "f_sym_ADVANC"


def main():
    rules = load_rules()
    assert sklearn.__version__ == "1.9.1" and np.__version__ == "2.3.5", "environment ไม่ตรง requirements-live.txt"
    assert rules["label"]["price_1630"] == "closing_auction" and rules["sessions"]["afternoon"]["cutoff"] == "12:35"
    ds = load_or_build_dataset(rules, rebuild=True)
    exps = load_experiments()
    fdh = next(v for v in exps["E12"]["variants"] if v["name"] == "fdh_lr")["features"]["afternoon"]
    flat = next(v for v in exps["E9FLAT"]["variants"] if v["name"] == "adir_lr")["features"]
    base = list(dict.fromkeys(list(fdh) + list(flat)))
    g = ds[ds["session"] == "afternoon"].sort_values(["date", "symbol"]).reset_index(drop=True)
    g[SYM_COL] = (g["symbol"] == "ADVANC").astype(float)
    cols = base + [SYM_COL]
    m, y = target_rows(g, "up_notup")
    g = g[m].reset_index(drop=True)
    X, dates, group = g[cols].to_numpy(float), g["date"].to_numpy(), g[SYM_COL].to_numpy()
    print(f"ข้อมูล: {g['date'].min().date()} – {g['date'].max().date()} · {len(g)} รอบบ่าย · สัดส่วนขึ้น {y.mean():.3f}", flush=True)
    res = tune("mlp", X, y, dates, group, seed=0)
    order = rank_features("mlp", X, y, dates, seed=0)
    use = order if res["k"] is None else order[:res["k"]]
    feature_cols = [cols[j] for j in use]
    model = build("mlp", res["params"], seed=0).fit(g[feature_cols], y)
    in_sample = proba(model, g[feature_cols])
    version = f"E15-mlp-{datetime.now():%Y%m%d%H%M}"
    out = OUTPUT_DIR / "models" / "R4ATC_UN" / "E15" / "mlp"
    out.mkdir(parents=True, exist_ok=True)
    bundle = {"kind": "sklearn", "model": model, "feature_cols": feature_cols, "group_by": ["session"], "variant": {"name": "mlp", "params": res["params"]},
              "model_version": version, "trained_until": str(g["date"].max().date()), "direction_target": "up_notup"}
    for sym in rules["symbols"]:
        path = out / f"{sym}_afternoon.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print("บันทึก", path)
    print(f"params {res['params']} · k {res['k']} · trial {res['n_trials']} · inner log loss {res['inner_ll']:.5f} · E0-un inner {res['e0_ll']:.5f}")
    print(f"feature {len(feature_cols)} ตัว · ในชุดฝึก P(ขึ้น) {in_sample.min():.3f}–{in_sample.max():.3f} · ทายขึ้น {(in_sample > .5).mean():.3%} · รุ่น {version}")


if __name__ == "__main__":
    main()
