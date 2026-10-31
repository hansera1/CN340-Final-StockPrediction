"""ทำนายหนึ่งรอบของวันจริง

1. แท่ง = ประวัติไฟล์หลัก + แท่ง 1m วันนี้ (Settrade → yfinance) รวมเป็น data.interval แบบ yfinance
   แล้วตัดทุกอย่างที่จบหลังจุดตัดทิ้ง (as_of = cutoff) → เห็นอนาคตไม่ได้ และไม่ใช้แท่งที่ยังไม่จบ
2. สร้าง features ด้วยฟังก์ชันเดียวกับ backtest
3. โหลด bundle ที่ล็อกแล้ว แล้วทำนาย
4. ถ้าขั้นใดล้ม → baseline จากข้อมูลล่าสุดที่ผ่านเกณฑ์ พร้อมธง fallback
"""
import pickle
import traceback
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_experiments, quantile_cols, session_times, to_local
from forecast.data.asof import last_close
from forecast.data.io import raw_path
from forecast.evaluation.metrics import round_to_tick
from forecast.evaluation.runner import design_matrix, sequences_for
from forecast.features.dataset import assemble_dataset
from forecast.labels.make_labels import labels_for
from forecast.live.log_schema import PredictionLog
from forecast.live.sources import live_bars
from forecast.models.baseline import BaselineModel


def model_dir(rules):
    live = rules["live"]
    return OUTPUT_DIR / "models" / live["experiment"] / live["variant"]


def live_variant(rules):
    live = rules["live"]
    exp = load_experiments()[live["experiment"]]
    return next(v for v in exp["variants"] if v["name"] == live["variant"])


def _fallback(bars, rules, sess, date):
    past = labels_for(bars, rules, rules["data"]["interval"])
    past = past[(past["session"] == sess) & (past["date"] < date)]
    m = BaselineModel(rules["quantiles"], target=rules["live"].get("fallback_target", "up")).fit(
        None, past["up"].to_numpy(), past["z"].to_numpy(), past)
    return m.predict(pd.DataFrame(index=[0])).iloc[0]


def predict_session(session, date, rules, raw_dir=None, now=None, source_dirs=None):
    date = to_local(date, rules).normalize()
    t = session_times(rules, session, date)
    now = to_local(now or datetime.now(), rules)
    as_of = min(t["cutoff"], now)
    mdir = model_dir(rules)

    by_sym, metas = {}, {}
    for sym in rules["symbols"]:
        b, metas[sym] = live_bars(sym, rules, date, as_of, raw_dir, source_dirs)
        by_sym[sym] = b[b["bar_end"] <= as_of].reset_index(drop=True)

    ds, err = None, None
    try:
        v = live_variant(rules)
        ds = assemble_dataset(rules, bars_by_symbol=by_sym, as_of=as_of, live_row=(date, session),
                              feature_minutes=v.get("feature_minutes"), spectral=v.get("spectral", False))
    except Exception:
        err = "features: " + traceback.format_exc(limit=1).strip().splitlines()[-1]

    records = []
    for sym in rules["symbols"]:
        bars, meta = by_sym[sym], metas[sym]
        # ราคาอ้างอิง = ราคาปิดของแท่งล่าสุดที่จบก่อนจุดตัด — นิยามเดียวกับ p_ref ใน labels ของ backtest
        p_ref, ref_time = last_close(bars, as_of)
        received = datetime.fromtimestamp(raw_path(sym, rules, raw_dir).stat().st_mtime)
        reason, version, fver = err, "baseline", "n/a"
        if session == "afternoon" and meta["source"] is None:
            reason = reason or "ไม่มีแท่งวันนี้จากทุกแหล่ง: " + "; ".join(meta["skipped"])

        if session == "afternoon" and pd.notna(ref_time):
            age = (as_of - ref_time).total_seconds() / 60
            if age > rules["live"]["max_price_age_min"]:
                reason = reason or f"ราคาอ้างอิงเก่า {age:.0f} นาที"

        pred = None
        if ds is not None:
            try:
                with open(mdir / f"{sym}_{session}.pkl", "rb") as f:
                    bundle = pickle.load(f)
                row = ds[(ds["symbol"] == sym) & (ds["session"] == session) & (ds["date"] == date)]
                X = design_matrix(row, [c for c in bundle["feature_cols"] if c in row], bundle["group_by"])
                X = X.reindex(columns=bundle["feature_cols"])
                X[[c for c in X if c.startswith("f_sym_")]] = X.filter(like="f_sym_").fillna(0.0)
                extra = (sequences_for(row.reset_index(drop=True), rules, raw_dir),) \
                    if getattr(bundle["model"], "needs_sequences", False) else ()
                pred = bundle["model"].predict(X, row, *extra).iloc[0]
                version = bundle["model_version"]
                fver = f"trained_until={bundle['trained_until']}"
            except Exception:
                reason = reason or "model: " + traceback.format_exc(limit=1).strip().splitlines()[-1]
        if pred is None:
            pred = _fallback(bars, rules, session, date)

        qcols = quantile_cols(rules)
        prices = {c: float(round_to_tick(p_ref * np.exp(pred[c]))) for c in qcols}
        records.append(PredictionLog(
            prediction_id=uuid.uuid4().hex[:12], symbol=sym, session=session, date=str(date.date()),
            data_cutoff=str(as_of), source_received_at=str(received), generated_at=str(datetime.now()),
            submitted_at=None, model_version=version, feature_version=fver,
            reference_price=p_ref, reference_time=str(ref_time),
            p_up=float(pred["p_up"]), direction="ขึ้น" if pred["p_up"] >= 0.5 else "ลง",
            z_quantiles={c: float(pred[c]) for c in qcols}, price_quantiles=prices,
            point_price=prices["q50"], fallback_reason=reason,
            data_source=meta["source"], data_lag_sec=meta["lag_sec"],
        ))
    return records
