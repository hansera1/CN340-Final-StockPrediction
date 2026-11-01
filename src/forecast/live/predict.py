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
3. Fallback ทุกขั้นเป็นโมเดล ML (รอบที่ 5 ข้อ A2) — ใช้กับทิศทาง ราคา และช่วงราคา:
   ขั้น 1 feature วันนี้ → ขั้น 2 feature จากข้อมูลถึงวันทำการก่อน (ล้มอีก → ไม่ใช้ไฟล์รายวัน)
   → ขั้น 3 แถว feature ว่าง (imputer เติมค่ามัธยฐานชุดฝึก) → ขั้น 4 E0-ud เฉพาะเมื่อโหลดไฟล์โมเดลไม่ได้
   บันทึกขั้นและ data_age_min ใน fallback_reason ทุกครั้ง
4. รันเงา (shadow): E0-ud + ราคาล่าสุด ณ จุดตัด — แยก log ห้ามส่ง ถ้าล้มระบบหลักทำงานต่อ
5. ข้อมูลรายวันของวันทำการก่อนไม่มา (live/daily_check.py): 05_predict_live.py ส่ง data_root = สำเนาข้อมูลที่เติมแถวสำรอง
   และ extra_reasons = ขั้นที่ใช้ → ต่อท้าย fallback_reason
6. live.price_consistency (ปิดไว้): ราคาจุดตามทิศของโมเดลทิศทางเมื่อมั่นใจพอ (ข้อ 1ง) — ใช้ทิศทางที่ส่งจริง
7. E13 ชั้น 2 รอบบ่าย (live.stage2, live/stage2_live.py): β ฝึกตอนเย็นวันก่อน · shadow = ส่ง p₁ บันทึก p_final
"""
import pickle
import traceback
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from forecast.config import EXTERNAL_DIR, OUTPUT_DIR, RAW_DIR, quantile_cols, session_times, to_local
from forecast.data.asof import last_close
from forecast.data.io import raw_path
from forecast.evaluation.metrics import round_to_tick, tick_size
from forecast.evaluation.runner import design_matrix
from forecast.features.dataset import assemble_dataset
from forecast.labels.make_labels import labels_for
from forecast.live.log_schema import PredictionLog, ShadowLog
from forecast.live.sources import live_bars
from forecast.live.stage2_live import apply as stage2_apply
from forecast.models.baseline import BaselineModel


def bundle_dir(spec):
    """'E9/lrx' → outputs/models/E9/lrx ('R4ATC/E9/lrx' → outputs/models/R4ATC/E9/lrx)"""
    return OUTPUT_DIR / "models" / spec


def _err(prefix):
    return f"{prefix}: " + traceback.format_exc(limit=1).strip().splitlines()[-1]


def load_bundle(spec, sym, session):
    with open(bundle_dir(spec) / f"{sym}_{session}.pkl", "rb") as f:
        return pickle.load(f)


def predict_row(bundle, sym, session, ds, date):
    """แถวผลทำนายของ bundle · ds = None → แถว feature ว่าง (ขั้น 3: imputer ของโมเดลเติมค่ามัธยฐานชุดฝึก)"""
    if ds is None:
        row = pd.DataFrame([{"date": date, "symbol": sym, "session": session, "sig20": np.nan, "p_ref": np.nan}])
        X = pd.DataFrame(np.nan, index=row.index, columns=bundle["feature_cols"])
        X[f"f_sym_{sym}"], X[f"f_sess_{session}"] = 1.0, 1.0
        X = X.reindex(columns=bundle["feature_cols"])
    else:
        row = ds[(ds["symbol"] == sym) & (ds["session"] == session) & (ds["date"] == date)]
        if row.empty:
            raise ValueError("ไม่มีแถว features ของรอบนี้")
        X = design_matrix(row, [c for c in bundle["feature_cols"] if c in row], bundle["group_by"])
        X = X.reindex(columns=bundle["feature_cols"])
    X[[c for c in X if c.startswith(("f_sym_", "f_sess_"))]] = X.filter(regex="^f_(sym|sess)_").fillna(0.0)
    return bundle["model"].predict(X, row).iloc[0]


def predict_bundle(spec, sym, session, ds, date):
    """→ (แถวผลทำนาย, bundle) ของโมเดลที่ล็อกไว้"""
    bundle = load_bundle(spec, sym, session)
    return predict_row(bundle, sym, session, ds, date), bundle


STEP_TH = {1: "ขั้น 1 feature วันนี้", 2: "ขั้น 2 feature จากข้อมูลถึงวันทำการก่อน", 3: "ขั้น 3 แถว feature ว่าง (imputer)",
           4: "ขั้น 4 E0-ud"}


def ml_predict(spec, sym, session, date, get_ds, what):
    """A2 (รอบที่ 5): ทุกขั้นเป็นโมเดลที่ล็อก → (แถวผล, bundle, ขั้น, เหตุผล)
    ขั้น 1 feature วันนี้ → ขั้น 2 feature จากข้อมูลล่าสุดที่มี → ขั้น 3 แถวว่าง · โหลดไฟล์โมเดลไม่ได้ → raise (ขั้น 4 ที่ผู้เรียก)"""
    bundle = load_bundle(spec, sym, session)
    notes = []
    for level in (1, 2):
        ds, err = get_ds(level)
        if ds is None:
            notes.append(f"{what} ขั้น {level}: {err}")
            continue
        try:
            return predict_row(bundle, sym, session, ds, date), bundle, level, notes
        except Exception:
            notes.append(_err(f"{what} ขั้น {level}"))
    return predict_row(bundle, sym, session, None, date), bundle, 3, notes


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


def xd_dividend(sym, date):
    """เงินปันผลต่อหุ้นถ้า date เป็นวัน XD ของหุ้นนี้ (data/external/calendar.csv) ไม่งั้น None"""
    path = EXTERNAL_DIR / "calendar.csv"
    if not path.exists():
        return None
    cal = pd.read_csv(path, parse_dates=["date"])
    m = (cal["event"] == "XD") & (cal["symbol"] == sym) & (cal["date"] == pd.Timestamp(date).tz_localize(None).normalize())
    return float(cal.loc[m, "amount"].sum()) if m.any() else None


def reference_price(sym, session, date, as_of, bars, meta, rules, daily_dir=None):
    """P(c) → (ราคา, เวลา, แหล่ง, เหตุผลถ้าใช้ทางสำรอง / ปรับ XD)"""
    if session == "morning":
        p, t, src = official_close(sym, date, rules, daily_dir)
        prev_day = bars.loc[bars["date"] < date, "date"].max() if len(bars) else pd.NaT
        if src and pd.notna(prev_day) and pd.Timestamp(t).normalize() == prev_day.tz_localize(None).normalize():
            div = xd_dividend(sym, date)
            if div:   # ข้อ 5: วัน XD → ราคาปิดเมื่อวาน − เงินปันผล ปัดช่องราคา
                adj = float(round_to_tick(p - div))
                return adj, t, f"{src} − XD", f"วัน XD: P(c) = {p:g} − ปันผล {div:g} = {adj:g}"
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


def cond_z_means(ds, sym, session, date):
    """E[z | ราคาปลาย > P(c)] และ E[z | < P(c)] จากประวัติที่รู้ผลแล้วของ (หุ้น, รอบ) เดียวกันก่อนวันที่ทาย"""
    h = ds[(ds["symbol"] == sym) & (ds["session"] == session) & (ds["date"] < date)]["z"].dropna()
    return float(h[h > 0].mean()), float(h[h < 0].mean())


def decimal_point(p_ref, p_hi, p_lo, z_up, z_dn):
    """รอบที่ 5 ข้อ B: ราคาไม่ปัดช่อง = P(c) · exp(P(สูงกว่า)·E[z|สูงกว่า] + P(ต่ำกว่า)·E[z|ต่ำกว่า])"""
    return p_ref * float(np.exp(p_hi * z_up + p_lo * z_dn))


def consistent_point(p_ref, rule_point, p_up, margin):
    """ข้อ 1ง: P(ขึ้น) > 0.5 + margin → P(c) + 1 ช่อง, < 0.5 − margin → P(c) − 1 ช่อง
    นอกนั้นใช้ราคาจากกฎค่ามัธยฐาน ยกเว้นกฎเลื่อนสวนทิศทาง → P(c)"""
    tick = float(tick_size(p_ref))
    if p_up > 0.5 + margin:
        return p_ref + tick
    if p_up < 0.5 - margin:
        return p_ref - tick
    against = (rule_point - p_ref) * (1 if p_up >= 0.5 else -1) < 0
    return p_ref if against else rule_point


def predict_session(session, date, rules, raw_dir=None, now=None, source_dirs=None, daily_dir=None,
                    data_root=None, extra_reasons=None):
    """→ (records ที่ส่งจริง, shadow records)
    data_root: โฟลเดอร์แทน data/raw (มี settrade/, yfinance/) — ใช้กับสำเนาข้อมูลสำรองและการซ้อม
    extra_reasons: {หุ้น: [ข้อความ]} ต่อท้าย fallback_reason (ขั้น fallback ของข้อมูลรายวัน)"""
    live = dict(rules["live"])
    if live.get("model_dir"):   # ชุดโมเดลสำรอง เช่น R4ATC (label 16:30 = ATC)
        for k in ["direction", "price_eq", "price_up", "shadow", "flat_dir"]:
            live[k] = f"{live['model_dir']}/{live[k]}"
    if data_root is not None:
        raw_dir = data_root / rules["data"]["source"]
        source_dirs = {s: data_root / s for s in live["sources"]}
        daily_dir = data_root
    date = to_local(date, rules).normalize()
    t = session_times(rules, session, date)
    now = to_local(now or datetime.now(), rules)
    as_of = min(t["cutoff"], now)
    qcols = quantile_cols(rules)

    by_sym, metas = {}, {}
    for sym in rules["symbols"]:
        b, metas[sym] = live_bars(sym, rules, date, as_of, raw_dir, source_dirs)
        by_sym[sym] = b[b["bar_end"] <= as_of].reset_index(drop=True)

    cache = {}

    def get_ds(level):
        """ขั้น 1 = แท่งถึงจุดตัด · ขั้น 2 = เฉพาะประวัติถึงวันทำการก่อน (ล้มอีก → ไม่ใช้ไฟล์รายวัน) → (ds, error)"""
        if level not in cache:
            if level == 1:
                tries = [(by_sym, daily_dir)]
            else:
                hist = {s: b[b["date"] < date].reset_index(drop=True) for s, b in by_sym.items()}
                tries = [(hist, daily_dir), (hist, (data_root or RAW_DIR) / "_none")]
            cache[level] = (None, "ไม่มีข้อมูล")
            for bars_l, dd in tries:
                try:
                    cache[level] = (assemble_dataset(rules, bars_by_symbol=bars_l, as_of=as_of,
                                                     live_row=(date, session), daily_dir=dd), None)
                    break
                except Exception:
                    cache[level] = (None, _err("features"))
        return cache[level]

    records, shadows = [], []
    for sym in rules["symbols"]:
        bars, meta = by_sym[sym], metas[sym]
        reasons = list((extra_reasons or {}).get(sym, []))
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

        # ทิศทาง: โมเดลที่ล็อก (ขั้น 1–3) → ขั้น 4 E0-ud เฉพาะเมื่อโหลดไฟล์โมเดลไม่ได้
        version, fver = "baseline E0-ud (fallback)", "n/a"
        try:
            pred, bundle, step, notes = ml_predict(live["direction"], sym, session, date, get_ds, "ทิศทาง")
            version, fver = bundle["model_version"], f"trained_until={bundle['trained_until']}"
        except Exception:
            pred, step, notes = _baseline(bars, rules, session, date), 4, [_err("ทิศทาง โหลดโมเดลไม่ได้")]
        reasons += notes
        if step > 1:
            reasons.append(f"ทิศทาง: {STEP_TH[step]}")
            if step == 2:
                hist_end = bars.loc[bars["date"] < date, "bar_end"].max()
                data_age_min = (as_of - hist_end).total_seconds() / 60 if pd.notna(hist_end) else None
        ds = get_ds(step)[0] if step < 3 else None
        p_up = float(pred["p_up"])
        # E13 ชั้น 2 (รอบบ่าย): shadow → ส่ง p₁ บันทึก p_final · live → ส่ง p_final · ล้มทุกกรณี → p_final = p₁
        p_stage1, p_final, s2_betas, s2_reason = p_up, None, None, None
        s2 = live.get("stage2") or {}
        if session == "afternoon" and s2.get("enabled"):
            try:
                p_final, s2_betas, s2_reason = stage2_apply(rules, date, p_up, meta.get("today_1m"))
            except Exception:
                p_final, s2_reason = p_up, _err("stage2")
            if s2.get("mode") == "live":
                p_up = p_final
                if s2_reason:
                    reasons.append(f"ชั้น 2: {s2_reason} → p_final = p₁")
        direction = "ขึ้น" if p_up >= 0.5 else "ลง"
        if (live.get("three_way") or {}).get("enabled") and step < 4:
            try:   # ข้อ 1ข: กฎ 3 คำตอบ
                f = float(predict_row(load_bundle(live["flat_dir"], sym, session), sym, session, ds, date)["p_up"])
                direction = ["ขึ้น", "เท่าเดิม", "ลง"][int(np.argmax([(1 - f) * p_up, f, (1 - f) * (1 - p_up)]))]
            except Exception:
                reasons.append(_err("flat model") + " → ทิศทางขึ้น/ลง")

        # ราคา: กฎค่ามัธยฐาน (ขั้น 1–3 เหมือนทิศทาง) → โหลดโมเดลราคาไม่ได้ ใช้ P(c)
        p_eq = p_price = p_hi = p_lo = None
        point = p_ref
        try:
            e, _, se, ne = ml_predict(live["price_eq"], sym, session, date, get_ds, "P(เท่า)")
            u, _, su, nu = ml_predict(live["price_up"], sym, session, date, get_ds, "p_price")
            p_eq, p_price = float(e["p_up"]), float(u["p_up"])
            point, p_hi, p_lo = median_rule(p_ref, p_eq, p_price)
            if max(se, su) > 1:
                reasons.append(f"ราคา: {STEP_TH[max(se, su)]}")
        except Exception:
            reasons.append(_err("price model") + " → ราคาจุด = P(c)")
        point_rule, decimal = point, False
        pc = live.get("price_consistency") or {}
        if (live.get("price_decimal") or {}).get("enabled"):   # ข้อ B: ส่งทศนิยมได้ (ไม่ใช้คู่กับ price_consistency)
            hist = get_ds(1)[0] if get_ds(1)[0] is not None else get_ds(2)[0]
            if p_hi is not None and hist is not None:
                point, decimal = decimal_point(p_ref, p_hi, p_lo, *cond_z_means(hist, sym, session, date)), True
            else:
                reasons.append("ราคาไม่ปัดช่องใช้ไม่ได้ (ไม่มีผลโมเดลราคา/ประวัติ) → ราคาปัดช่องจากกฎค่ามัธยฐาน")
        elif pc.get("enabled"):
            point = consistent_point(p_ref, point, p_up, pc["margin"])

        # ช่วงราคา: quantile ของ z (E0) รอบราคาจุด — ลำดับขั้นเดียวกัน
        try:
            zq = ml_predict(live["shadow"], sym, session, date, get_ds, "ช่วงราคา")[0]
        except Exception:
            zq = _baseline(bars, rules, session, date)
            reasons.append("ช่วงราคา: ขั้น 4 E0-ud จากประวัติ")
        rnd = (lambda x: round(float(x), 2)) if decimal else (lambda x: float(round_to_tick(x)))
        prices = {c: rnd(point * np.exp(zq[c] - zq["q50"])) for c in qcols}
        prices["q50"] = rnd(point)

        records.append(PredictionLog(
            prediction_id=uuid.uuid4().hex[:12], symbol=sym, session=session, date=str(date.date()),
            data_cutoff=str(as_of), source_received_at=str(datetime.fromtimestamp(
                raw_path(sym, rules, raw_dir).stat().st_mtime)),
            generated_at=str(datetime.now()), submitted_at=None, model_version=version, feature_version=fver,
            reference_price=p_ref, reference_time=str(ref_time), reference_source=ref_src,
            p_up=p_up, direction=direction,
            z_quantiles={c: float(np.log(prices[c] / p_ref)) for c in qcols}, price_quantiles=prices,
            point_price=prices["q50"], point_rule=float(round_to_tick(point_rule)), p_eq=p_eq, p_price=p_price, p_higher=p_hi, p_lower=p_lo,
            fallback_reason="; ".join(dict.fromkeys(reasons)) or None,
            p_stage1=p_stage1, p_final=p_final, stage2_betas=s2_betas, stage2_reason=s2_reason,
            data_source=meta["source"], data_lag_sec=meta["lag_sec"], data_age_min=data_age_min,
        ))

        # รันเงา: E0-ud + ราคาล่าสุด ณ จุดตัด — ล้มได้โดยไม่กระทบระบบหลัก
        try:
            sp = (predict_row(load_bundle(live["shadow"], sym, session), sym, session, ds, date) if step < 4
                  else _baseline(bars, rules, session, date))
            shadows.append(ShadowLog(
                symbol=sym, session=session, date=str(date.date()), data_cutoff=str(as_of),
                generated_at=str(datetime.now()), model="E0-ud + ราคาล่าสุด ณ จุดตัด",
                p_up=float(sp["p_up"]), direction="ขึ้น" if sp["p_up"] >= 0.5 else "ลง",
                reference_price=p_ref, point_price=float(round_to_tick(p_ref)),
                price_quantiles={c: float(round_to_tick(p_ref * np.exp(sp[c]))) for c in qcols}))
        except Exception:
            pass
    return records, shadows
