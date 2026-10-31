"""รันการทดลอง E0–E6 แบบ walk-forward และฝึกโมเดลสุดท้ายสำหรับวันจริง"""
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from forecast.config import PROCESSED_DIR
from forecast.data.io import load_bars
from forecast.evaluation.splits import holdout_split, walk_forward_splits
from forecast.features.dataset import add_flat_features, add_sigma, assemble_dataset, select_features
from forecast.features.sequences import build_sequences
from forecast.models import get_model

KEEP = ["date", "symbol", "session", "p_ref", "p_end", "z", "r", "direction", "up"]


def experiment_rules(rules, exp):
    """ค่า evaluation ของการทดลองนั้น (เช่น pilot ที่ข้อมูลน้อยกว่า) ทับค่ากลาง"""
    r = {**rules, "evaluation": {**rules["evaluation"], **exp.get("evaluation", {})}}
    return r


def variant_interval(rules, exp, variant):
    return variant.get("interval") or exp.get("interval") or rules["data"]["interval"]


def load_or_build_dataset(rules, feature_minutes=None, spectral=False, raw_dir=None, rebuild=False,
                          interval=None):
    interval = interval or rules["data"]["interval"]
    # แยก cache ตามแหล่ง/ความละเอียด (ข้อมูลจำลองไม่ปนข้อมูลจริง)
    source = raw_dir.name.strip("_") if raw_dir else rules["data"]["source"]
    fm = f"_f{feature_minutes}m" if feature_minutes else ""
    name = f"dataset_{source}_{interval}{fm}{'_spectral' if spectral else ''}.pkl"
    path = PROCESSED_DIR / name
    if path.exists() and not rebuild:
        ds = pd.read_pickle(path)
        if "sig20" not in ds:                     # cache รุ่นก่อนรอบที่ 2
            ds = add_sigma(ds)
        if "fl_exp_ticks" not in ds:              # cache รุ่นก่อน E9-FLAT
            bars = {s: load_bars(s, rules, raw_dir, interval) for s in rules["symbols"]}
            ds = add_flat_features(ds, bars, rules)
        return ds.reset_index(drop=True)
    ds = assemble_dataset(rules, raw_dir=raw_dir, feature_minutes=feature_minutes, spectral=spectral,
                          interval=interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_pickle(path)
    return ds


def overlap_dates(rules, intervals, raw_dir=None):
    """วันที่มีข้อมูลครบทุกความละเอียดและทุกหุ้น (E3 ต้องเทียบบนช่วงวันเดียวกัน)"""
    common = None
    for iv in intervals:
        for sym in rules["symbols"]:
            d = set(load_bars(sym, rules, raw_dir, iv)["date"])
            common = d if common is None else common & d
    return sorted(common or [])


def design_matrix(g, fcols, group_by):
    X = g[fcols].astype(float)
    if "symbol" not in group_by:   # แชร์ข้อมูลข้ามหุ้น: ใส่ตัวแปรบอกหุ้น
        X = X.join(pd.get_dummies(g["symbol"], prefix="f_sym", dtype=float))
    if "session" not in group_by:  # รวมสองรอบในโมเดลเดียว (E9-FLAT Model A): ใส่ตัวแปรบอกรอบ
        X = X.join(pd.get_dummies(g["session"], prefix="f_sess", dtype=float))
    return X


def sequences_for(ds, rules, raw_dir=None, interval=None):
    seqs = np.zeros((len(ds), 3, 120), dtype=np.float32)
    for sym in ds["symbol"].unique():
        m = (ds["symbol"] == sym).to_numpy()
        seqs[m] = build_sequences(load_bars(sym, rules, raw_dir, interval), ds[m], rules)
    return seqs


def embeddings_for(ds, rules, kind, raw_dir=None, interval=None, length=56):
    """E7: embedding ResNet18 ต่อแถวของ ds (ลำดับเดียวกับ ds) — cache ไว้ใน data/processed/"""
    interval = interval or rules["data"]["interval"]
    source = raw_dir.name.strip("_") if raw_dir else rules["data"]["source"]
    path = PROCESSED_DIR / f"emb_{kind}_{source}_{interval}_L{length}.npz"
    keys = ds["date"].astype(str) + "|" + ds["symbol"] + "|" + ds["session"]
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if len(z["keys"]) == len(keys) and (z["keys"] == keys.to_numpy().astype(str)).all():
            return z["emb"]
    from forecast.models.resnet_embed import embed_rows
    emb = np.zeros((len(ds), 512), dtype=np.float32)
    for sym in ds["symbol"].unique():
        m = (ds["symbol"] == sym).to_numpy()
        emb[m] = embed_rows(load_bars(sym, rules, raw_dir, interval), ds[m], rules, kind, length)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, emb=emb, keys=keys.to_numpy().astype(str))
    return emb


def extra_dataset(rules, symbols, raw_dir=None, interval=None):
    """E10-20: labels (+ sig20) ของหุ้นเพิ่มเติม ใช้เป็นข้อมูลฝึกเท่านั้น — สร้างแบบเดียวกับหุ้นเป้าหมาย
    (P(10:00) = open รายวันของ Settrade) แต่ไม่มี features ตาราง"""
    from forecast.labels.make_labels import labels_for
    interval = interval or rules["data"]["interval"]
    frames = [labels_for(load_bars(sym, rules, raw_dir, interval), rules, interval) for sym in symbols]
    return add_sigma(pd.concat(frames, ignore_index=True)).reset_index(drop=True)


def images_for(ds, rules, kind, raw_dir=None, interval=None, length=56, tag=""):
    """E10: ภาพ 0–255 (uint8) ต่อแถวของ ds — normalize/resize ตอนฝึกตาม arch; cache ใน data/processed/"""
    interval = interval or rules["data"]["interval"]
    source = raw_dir.name.strip("_") if raw_dir else rules["data"]["source"]
    path = PROCESSED_DIR / f"img_{kind}_{source}_{interval}_L{length}{tag}.npz"
    keys = ds["date"].astype(str) + "|" + ds["symbol"] + "|" + ds["session"]
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if len(z["keys"]) == len(keys) and (z["keys"] == keys.to_numpy().astype(str)).all():
            return z["img"]
    from forecast.features.images import build_images
    img = np.zeros((len(ds), 3, 224, 224), dtype=np.uint8)
    for sym in ds["symbol"].unique():
        m = np.flatnonzero((ds["symbol"] == sym).to_numpy())
        bars = load_bars(sym, rules, raw_dir, interval)
        for s in range(0, len(m), 256):
            x = build_images(bars, ds.iloc[m[s:s + 256]], rules, kind, length, normalize=False)
            img[m[s:s + 256]] = np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, img=img, keys=keys.to_numpy().astype(str))
    return img


def run_variant(ds, variant, rules, seqs=None, holdout_eval=False, test_symbols=None):
    """test_symbols: ทดสอบเฉพาะหุ้นเหล่านี้ (E10-20: หุ้นอื่นอยู่ในชุดฝึกอย่างเดียว)"""
    ev = rules["evaluation"]
    group_by = variant.get("group_by", ["symbol", "session"])
    d = ds[ds["session"].isin(variant["sessions"])] if variant.get("sessions") else ds
    preds = []
    for _, g in (d.groupby(group_by) if group_by else [((), d)]):
        X = design_matrix(g, select_features(g, variant), group_by)
        dates = g["date"].to_numpy()
        labeled = g["up"].notna().to_numpy()
        splits = (holdout_split(dates, ev["holdout_days"]) if holdout_eval else
                  walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]))
        for tr_dates, te_dates in splits:
            tr = np.isin(dates, tr_dates) & labeled
            te = np.isin(dates, te_dates)
            if test_symbols is not None:
                te &= g["symbol"].isin(test_symbols).to_numpy()
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
    version = f"{exp_code}-{variant['name']}-{datetime.now():%Y%m%d%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, g in (d.groupby(group_by) if group_by else [((), d)]):
        g = g[g["up"].notna()]
        X = design_matrix(g, select_features(g, variant), group_by)
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
            for sess in ([key["session"]] if "session" in key else list(rules["sessions"])):
                path = out_dir / f"{sym}_{sess}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(bundle, f)
                paths.append(path)
    return paths
