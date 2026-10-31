"""ทำนายหนึ่งรอบของวันจริง — เฉพาะโมเดลที่ล็อก (NEXT_STEPS รอบที่ 3 ข้อ 3, 7)

ทิศทาง  E12 fd_lr = F0 + F-daily, LR สเปก E9-LR (rules.yaml → live.direction; results/lock_decision.md)
ราคา    กฎค่ามัธยฐาน: A-price LR ทาย P(ราคาปลาย = P(c)), p_price ทาย P(ราคาปลาย > P(c) | ไม่เท่า)
        P(สูงกว่า) = (1 − P(เท่า)) × p_price → ราคาจุด = P(c) + 1 ช่องถ้า > 0.5, − 1 ช่องถ้า P(ต่ำกว่า) > 0.5, นอกนั้น P(c)
        ช่วงราคา = quantile เชิงประจักษ์ของ z (E0) รอบราคาจุด
P(c)    รอบบ่าย = ราคาซื้อขายล่าสุดจากแท่ง 1m ณ จุดตัด · รอบเช้า = ราคาปิดทางการ (รวม ATC) ของวันทำการก่อน (Settrade 1d)
        features ยังคำนวณจากแท่ง 60m เหมือน backtest

1. แท่ง = ประวัติไฟล์หลัก + แท่ง 1m วันนี้ (Settrade → yfinance) รวมเป็น data.interval แบบ yfinance
   แล้วตัดทุกอย่างที่จบหลังจุดตัดทิ้ง (as_of = cutoff) → เห็นอนาคตไม่ได้ และไม่ใช้แท่งที่ยังไม่จบ
2. สร้าง features ด้วยฟังก์ชันเดียวกับ backtest
3. Fallback: ข้อมูลวันนี้ไม่ทัน → ยังรันโมเดลทิศทาง ด้วย features จากข้อมูลล่าสุดที่ผ่านเกณฑ์ (ระบุอายุข้อมูล)
   baseline E0-ud เป็นทางสุดท้ายเมื่อโมเดลรันไม่ได้เลย · บันทึก fallback_reason ทุกครั้ง
4. รันเงา (shadow): E0-ud + ราคาล่าสุด ณ จุดตัด — แยก log ห้ามส่ง ถ้าล้มระบบหลักทำงานต่อ
"""
import pickle
import traceback
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, RAW_DIR, quantile_cols, session_times, to_local
from forecast.data.asof import last_close
from forecast.data.io import raw_path
from forecast.evaluation.metrics import round_to_tick, tick_size
from forecast.evaluation.runner import design_matrix
from forecast.features.dataset import assemble_dataset
from forecast.labels.make_labels import labels_for
from forecast.live.log_schema import PredictionLog, ShadowLog
from forecast.live.sources import live_bars
from forecast.models.baseline import BaselineModel


def bundle_dir(spec):
    """'E9/lrx' → outputs/models/E9/lrx"""
    return OUTPUT_DIR / "models" / spec


def _err(prefix):
    return f"{prefix}: " + traceback.format_exc(limit=1).strip().splitlines()[-1]


def predict_bundle(spec, sym, session, ds, date):
    """→ (แถวผลทำนาย, bundle) ของโมเดลที่ล็อกไว้"""
    with open(bundle_dir(spec) / f"{sym}_{session}.pkl", "rb") as f:
        bundle = pickle.load(f)
    row = ds[(ds["symbol"] == sym) & (ds["session"] == session) & (ds["date"] == date)]
    if row.empty:
        raise ValueError("ไม่มีแถว features ของรอบนี้")
    X = design_matrix(row, [c for c in bundle["feature_cols"] if c in row], bundle["group_by"])
    X = X.reindex(columns=bundle["feature_cols"])
    X[[c for c in X if c.startswith(("f_sym_", "f_sess_"))]] = X.filter(regex="^f_(sym|sess)_").fillna(0.0)
    return bundle["model"].predict(X, row).iloc[0], bundle


def _baseline(bars, rules, sess, date):
    """ทางสุดท้าย: E0-ud จากประวัติที่มี (ไม่ต้องใช้ไฟล์โมเดล)"""
    past = labels_for(bars, rules, rules["data"]["interval"])
    past = past[(past["session"] == sess) & (past["date"] < date)]
    m = BaselineModel(rules["quantiles"], target="updown").fit(None, past["up"].to_numpy(), past["z"].to_numpy(), past)
    return m.predict(pd.DataFrame(index=[0])).iloc[0]


def official_close(sym, date, rules, daily_dir=None):
    """ราคาปิดทางการ (รวม ATC) ของวันทำการล่าสุดก่อน date → (ราคา, วันที่, แหล่ง)"""
    base = daily_dir or RAW_DIR
    for src in ["settrade", "yfinance"]:
        path = base / src / f"{sym}_1d.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, parse_dates=["timestamp"])
        d = d[d["timestamp"].dt.normalize() < date.tz_localize(None).normalize()]
        if len(d):
            last = d.sort_values("timestamp").iloc[-1]
            return float(last["close"]), last["timestamp"], f"{src} 1d close"
    return np.nan, pd.NaT, None


def reference_price(sym, session, date, as_of, bars, meta, rules, daily_dir=None):
    """P(c) → (ราคา, เวลา, แหล่ง, เหตุผลถ้าใช้ทางสำรอง)"""
    if session == "morning":
        p, t, src = official_close(sym, date, rules, daily_dir)
        prev_day = bars.loc[bars["date"] < date, "date"].max() if len(bars) else pd.NaT
        if src and pd.notna(prev_day) and pd.Timestamp(t).normalize() == prev_day.tz_localize(None).normalize():
            return p, t, src, None
        p60, t60 = last_close(bars, as_of)
        why = "ไม่มีราคาปิดทางการของวันทำการก่อน" if not src else f"ราคาปิดทางการล่าสุดเป็นของ {pd.Timestamp(t):%d/%m}"
        return p60, t60, "60m last close", why + " → ใช้ราคาปิดแท่ง 60m ล่าสุด"
    today = meta.get("today_1m")
    if today is not None and len(today):
        p, t = last_close(today.sort_values("bar_end"), as_of)
        if np.isfinite(p):
            return p, t, f"{meta['source']} 1m", None
    p60, t60 = last_close(bars, as_of)
    return p60, t60, "60m last close", "ไม่มีแท่ง 1m วันนี้ → ใช้ราคาปิดแท่ง 60m ล่าสุด"


def median_rule(p_ref, p_eq, p_price):
    """→ (ราคาจุด, P(สูงกว่า), P(ต่ำกว่า))"""
    up, dn = (1 - p_eq) * p_price, (1 - p_eq) * (1 - p_price)
    tick = float(tick_size(p_ref))
    shift = 1 if up > 0.5 else -1 if dn > 0.5 else 0
    return p_ref + shift * tick, up, dn


def predict_session(session, date, rules, raw_dir=None, now=None, source_dirs=None, daily_dir=None):
    """→ (records ที่ส่งจริง, shadow records)"""
    live = rules["live"]
    date = to_local(date, rules).normalize()
    t = session_times(rules, session, date)
    now = to_local(now or datetime.now(), rules)
    as_of = min(t["cutoff"], now)
    qcols = quantile_cols(rules)

    by_sym, metas = {}, {}
    for sym in rules["symbols"]:
        b, metas[sym] = live_bars(sym, rules, date, as_of, raw_dir, source_dirs)
        by_sym[sym] = b[b["bar_end"] <= as_of].reset_index(drop=True)

    ds, feat_err = None, None
    try:
        ds = assemble_dataset(rules, bars_by_symbol=by_sym, as_of=as_of, live_row=(date, session))
    except Exception:
        feat_err = _err("features")

    records, shadows = [], []
    for sym in rules["symbols"]:
        bars, meta = by_sym[sym], metas[sym]
        reasons = [feat_err] if feat_err else []
        p_ref, ref_time, ref_src, why = reference_price(sym, session, date, as_of, bars, meta, rules, daily_dir)
        if why:
            reasons.append(why)
        last_bar = bars["bar_end"].max() if len(bars) else pd.NaT
        data_age_min = (as_of - last_bar).total_seconds() / 60 if pd.notna(last_bar) else None
        _, daily_t, daily_src = official_close(sym, date, rules, daily_dir)
        prev_day = bars.loc[bars["date"] < date, "date"].max() if len(bars) else pd.NaT
        if pd.notna(prev_day) and (not daily_src or pd.Timestamp(daily_t).normalize() <
                                   prev_day.tz_localize(None).normalize()):
            reasons.append("แท่งรายวันยังไม่มีของวันทำการก่อน → F-daily ใช้ข้อมูลรายวันที่เก่ากว่า")
        if session == "afternoon" and meta["source"] is None:
            reasons.append("ไม่มีแท่งวันนี้จากทุกแหล่ง (" + "; ".join(meta["skipped"]) +
                           f") → โมเดลใช้ข้อมูลล่าสุดที่ผ่านเกณฑ์ อายุ {data_age_min:.0f} นาที")
        if session == "afternoon" and pd.notna(ref_time):
            age = (as_of - ref_time).total_seconds() / 60
            if age > live["max_price_age_min"]:
                reasons.append(f"ราคาอ้างอิงเก่า {age:.0f} นาที")

        # ทิศทาง: โมเดลที่ล็อก → ทางสุดท้าย E0-ud
        pred, version, fver = None, "baseline E0-ud (fallback)", "n/a"
        if ds is not None:
            try:
                pred, bundle = predict_bundle(live["direction"], sym, session, ds, date)
                version, fver = bundle["model_version"], f"trained_until={bundle['trained_until']}"
            except Exception:
                reasons.append(_err("direction model"))
        if pred is None:
            pred = _baseline(bars, rules, session, date)
        p_up = float(pred["p_up"])

        # ราคา: กฎค่ามัธยฐาน → ถ้าโมเดลราคาล้ม ใช้ P(c)
        p_eq = p_price = p_hi = p_lo = None
        point = p_ref
        if ds is not None:
            try:
                p_eq = float(predict_bundle(live["price_eq"], sym, session, ds, date)[0]["p_up"])
                p_price = float(predict_bundle(live["price_up"], sym, session, ds, date)[0]["p_up"])
                point, p_hi, p_lo = median_rule(p_ref, p_eq, p_price)
            except Exception:
                reasons.append(_err("price model") + " → ราคาจุด = P(c)")

        # ช่วงราคา: quantile ของ z (E0) รอบราคาจุด
        try:
            zq = predict_bundle(live["shadow"], sym, session, ds, date)[0] if ds is not None else None
        except Exception:
            zq = None
        if zq is None:
            zq = _baseline(bars, rules, session, date)
        prices = {c: float(round_to_tick(point * np.exp(zq[c] - zq["q50"]))) for c in qcols}
        prices["q50"] = float(round_to_tick(point))

        records.append(PredictionLog(
            prediction_id=uuid.uuid4().hex[:12], symbol=sym, session=session, date=str(date.date()),
            data_cutoff=str(as_of), source_received_at=str(datetime.fromtimestamp(
                raw_path(sym, rules, raw_dir).stat().st_mtime)),
            generated_at=str(datetime.now()), submitted_at=None, model_version=version, feature_version=fver,
            reference_price=p_ref, reference_time=str(ref_time), reference_source=ref_src,
            p_up=p_up, direction="ขึ้น" if p_up >= 0.5 else "ลง",
            z_quantiles={c: float(np.log(prices[c] / p_ref)) for c in qcols}, price_quantiles=prices,
            point_price=prices["q50"], p_eq=p_eq, p_price=p_price, p_higher=p_hi, p_lower=p_lo,
            fallback_reason="; ".join(reasons) or None,
            data_source=meta["source"], data_lag_sec=meta["lag_sec"], data_age_min=data_age_min,
        ))

        # รันเงา: E0-ud + ราคาล่าสุด ณ จุดตัด — ล้มได้โดยไม่กระทบระบบหลัก
        try:
            sp = predict_bundle(live["shadow"], sym, session, ds, date)[0] if ds is not None else \
                _baseline(bars, rules, session, date)
            shadows.append(ShadowLog(
                symbol=sym, session=session, date=str(date.date()), data_cutoff=str(as_of),
                generated_at=str(datetime.now()), model="E0-ud + ราคาล่าสุด ณ จุดตัด",
                p_up=float(sp["p_up"]), direction="ขึ้น" if sp["p_up"] >= 0.5 else "ลง",
                reference_price=p_ref, point_price=float(round_to_tick(p_ref)),
                price_quantiles={c: float(round_to_tick(p_ref * np.exp(sp[c]))) for c in qcols}))
        except Exception:
            pass
    return records, shadows
