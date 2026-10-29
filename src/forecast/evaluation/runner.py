"""รันการทดลอง E0–E6 แบบ walk-forward และฝึกโมเดลสุดท้ายสำหรับวันจริง"""
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from forecast.config import PROCESSED_DIR
from forecast.data.io import load_bars
from forecast.evaluation.splits import holdout_split, walk_forward_splits
from forecast.features.dataset import assemble_dataset, feature_cols
from forecast.features.sequences import build_sequences
from forecast.models import get_model

KEEP = ["date", "symbol", "session", "p_ref", "p_end", "z", "r", "direction", "up"]


def load_or_build_dataset(rules, feature_minutes=1, spectral=False, raw_dir=None, rebuild=False):
    source = f"_{raw_dir.name.strip('_')}" if raw_dir else ""   # แยก cache ของข้อมูลจำลองออกจากข้อมูลจริง
    name = f"dataset{source}_{feature_minutes}m{'_spectral' if spectral else ''}.pkl"
    path = PROCESSED_DIR / name
    if path.exists() and not rebuild:
        return pd.read_pickle(path)
    ds = assemble_dataset(rules, raw_dir=raw_dir, feature_minutes=feature_minutes, spectral=spectral)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_pickle(path)
    return ds


def design_matrix(g, fcols, group_by):
    X = g[fcols].astype(float)
    if "symbol" not in group_by:   # แชร์ข้อมูลข้ามหุ้น: ใส่ตัวแปรบอกหุ้น
        X = X.join(pd.get_dummies(g["symbol"], prefix="f_sym", dtype=float))
    return X


def sequences_for(ds, rules, raw_dir=None):
    seqs = np.zeros((len(ds), 3, 120), dtype=np.float32)
    for sym in ds["symbol"].unique():
        m = (ds["symbol"] == sym).to_numpy()
        seqs[m] = build_sequences(load_bars(sym, rules, raw_dir), ds[m], rules)
    return seqs


def run_variant(ds, variant, rules, seqs=None, holdout_eval=False):
    ev = rules["evaluation"]
    group_by = variant.get("group_by", ["symbol", "session"])
    d = ds[ds["session"].isin(variant["sessions"])] if variant.get("sessions") else ds
    fcols = feature_cols(d)
    preds = []
    for _, g in d.groupby(group_by):
        X = design_matrix(g, fcols, group_by)
        dates = g["date"].to_numpy()
        labeled = g["up"].notna().to_numpy()
        splits = (holdout_split(dates, ev["holdout_days"]) if holdout_eval else
                  walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]))
        for tr_dates, te_dates in splits:
            tr = np.isin(dates, tr_dates) & labeled
            te = np.isin(dates, te_dates)
            model = get_model(variant["model"], rules["quantiles"], **variant.get("params", {}))
            extra_tr = (seqs[g.index[tr]],) if getattr(model, "needs_sequences", False) else ()
            extra_te = (seqs[g.index[te]],) if extra_tr else ()
            model.fit(X[tr], g["up"].to_numpy()[tr], g["z"].to_numpy()[tr], g[tr], *extra_tr)
            p = model.predict(X[te], g[te], *extra_te)
            preds.append(g.loc[te, KEEP].join(p))
    if not preds:
        raise ValueError("ข้อมูลไม่พอสำหรับ walk-forward — ลด min_train_days ใน rules.yaml หรือเพิ่มข้อมูล")
    return pd.concat(preds).sort_values(["date", "symbol", "session"]).reset_index(drop=True)


def fit_final(ds, variant, rules, exp_code, out_dir, seqs=None):
    """ฝึกบนข้อมูลที่มี label ทั้งหมด แล้วบันทึกเป็น bundle ต่อ (หุ้น, รอบ) สำหรับวันจริง"""
    group_by = variant.get("group_by", ["symbol", "session"])
    d = ds[ds["session"].isin(variant["sessions"])] if variant.get("sessions") else ds
    fcols = feature_cols(d)
    version = f"{exp_code}-{variant['name']}-{datetime.now():%Y%m%d%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, g in d.groupby(group_by):
        g = g[g["up"].notna()]
        X = design_matrix(g, fcols, group_by)
        model = get_model(variant["model"], rules["quantiles"], **variant.get("params", {}))
        extra = (seqs[g.index],) if getattr(model, "needs_sequences", False) else ()
        model.fit(X, g["up"].to_numpy(), g["z"].to_numpy(), g, *extra)
        key = dict(zip(group_by, np.atleast_1d(key)))
        bundle = {
            "model": model, "feature_cols": list(X.columns), "group_by": group_by,
            "variant": variant, "model_version": version,
            "trained_until": str(g["date"].max().date()),
        }
        for sym in ([key["symbol"]] if "symbol" in key else rules["symbols"]):
            path = out_dir / f"{sym}_{key['session']}.pkl"
            with open(path, "wb") as f:
                pickle.dump(bundle, f)
            paths.append(path)
    return paths
