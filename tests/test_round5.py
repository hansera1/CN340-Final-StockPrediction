"""รอบที่ 5: fallback ทุกขั้นเป็นโมเดล ML (A2), feature ชั้น 1 เท่ากันทุกจุดตัด 12:00–12:29 (A3), ร่างรอบเช้า (A5)
ทดสอบกับข้อมูลจริงและโมเดลที่ล็อก — ข้ามถ้าไม่มี"""
import copy

import numpy as np
import pandas as pd
import pytest

import forecast.live.predict as P
from forecast.config import OUTPUT_DIR, RAW_DIR, load_rules, session_times, to_local
from forecast.features.dataset import assemble_dataset
from forecast.live.sources import live_bars

REAL = (RAW_DIR / "settrade" / "KBANK_1m.csv").exists() and (OUTPUT_DIR / "models" / "E12" / "fd_lr").exists()
DAY, NOW = "2026-10-30", "2026-10-30 12:15"
pytestmark = pytest.mark.skipif(not REAL, reason="ไม่มีข้อมูลจริง / โมเดลที่ล็อก")


def _run():
    return P.predict_session("afternoon", DAY, load_rules(), now=NOW)[0]


def test_assemble_fails_uses_locked_model(monkeypatch):
    """assemble_dataset ล้มทุกครั้ง → ขั้น 3 (แถวว่าง) ด้วยโมเดลที่ล็อก ไม่ใช่ baseline · ยังได้ราคาและช่วงราคา"""
    def boom(*a, **k):
        raise RuntimeError("จำลอง assemble_dataset ล้ม")
    monkeypatch.setattr(P, "assemble_dataset", boom)
    recs = _run()
    assert len(recs) == 2
    for r in recs:
        assert r.model_version != "baseline E0-ud (fallback)"
        assert "ขั้น 3" in r.fallback_reason
        assert np.isfinite(r.p_up) and r.p_eq is not None
        assert all(np.isfinite(v) for v in r.price_quantiles.values())


def test_level1_predict_error_falls_to_level2(monkeypatch):
    """predict ขั้น 1 ล้ม → ขั้น 2 (feature จากประวัติถึงวันทำการก่อน) พร้อม data_age_min ของประวัติ"""
    real, first = P.predict_row, {}

    def flaky(bundle, sym, session, ds, date):
        if ds is not None and first.setdefault("ds", id(ds)) == id(ds):
            raise RuntimeError("จำลอง predict ขั้น 1 ล้ม")
        return real(bundle, sym, session, ds, date)
    monkeypatch.setattr(P, "predict_row", flaky)
    for r in _run():
        assert r.model_version != "baseline E0-ud (fallback)"
        assert "ทิศทาง: ขั้น 2" in r.fallback_reason
        assert r.data_age_min > 60          # ใช้ข้อมูลถึงวันทำการก่อน


def test_missing_model_files_level4(monkeypatch, tmp_path):
    """โหลดไฟล์โมเดลไม่ได้เลย → ขั้น 4 E0-ud พร้อมเหตุผล และยังได้ผลครบ"""
    monkeypatch.setattr(P, "bundle_dir", lambda spec: tmp_path / spec)
    for r in _run():
        assert r.model_version == "baseline E0-ud (fallback)"
        assert "โหลดโมเดลไม่ได้" in r.fallback_reason and "ขั้น 4" in r.fallback_reason
        assert all(np.isfinite(v) for v in r.price_quantiles.values())


def _stage1_row(rules, cut):
    rules = copy.deepcopy(rules)
    rules["sessions"]["afternoon"]["cutoff"] = cut
    date = to_local(DAY, rules).normalize()
    as_of = session_times(rules, "afternoon", date)["cutoff"]
    by = {}
    for s in rules["symbols"]:
        b, _ = live_bars(s, rules, date, as_of)
        by[s] = b[b["bar_end"] <= as_of].reset_index(drop=True)
    ds = assemble_dataset(rules, bars_by_symbol=by, as_of=as_of, live_row=(date, "afternoon"))
    return ds[(ds["date"] == date) & (ds["session"] == "afternoon")].set_index("symbol")


def test_stage1_features_same_for_cutoffs_1205_to_1215():
    """A3: แท่ง 60m 12:00–12:30 ยังไม่จบที่ 12:05–12:29 → feature ชั้น 1 ทุกตัวเท่ากัน (ต่างแค่ P(c) จาก 1m)"""
    rules = load_rules()
    rows = {c: _stage1_row(rules, c) for c in ["12:05", "12:10", "12:15"]}
    cols = [c for c in rows["12:15"] if c.startswith(("f_", "fd_", "fh_", "fl_"))]
    for c in ["12:05", "12:10"]:
        a, b = rows[c][cols].astype(float), rows["12:15"][cols].astype(float)
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), c


def test_draft_evening_equals_morning(tmp_path):
    """A5: ร่างจากข้อมูลที่มี ณ เย็น 29 ต.ค. (ตัดทุกแถวตั้งแต่ 30 ต.ค.) = ผลรันเช้า 30 ต.ค. บนข้อมูลเต็ม ทุกค่า"""
    import shutil
    rules = load_rules()
    for src in ["settrade", "yfinance"]:
        (tmp_path / src).mkdir()
        for sym in rules["symbols"]:
            for f in (RAW_DIR / src).glob(f"{sym}_*.csv"):
                d = pd.read_csv(f)
                d[pd.to_datetime(d["timestamp"]) < pd.Timestamp(DAY)].to_csv(tmp_path / src / f.name, index=False)
    now = f"{DAY} 09:15"
    draft = P.predict_session("morning", DAY, rules, now=now, data_root=tmp_path)[0]
    morning = P.predict_session("morning", DAY, rules, now=now)[0]
    for a, b in zip(draft, morning):
        assert a.symbol == b.symbol and a.direction == b.direction
        assert a.p_up == pytest.approx(b.p_up, abs=1e-12) and a.p_eq == pytest.approx(b.p_eq, abs=1e-12)
        assert a.price_quantiles == b.price_quantiles and a.reference_price == b.reference_price
        assert a.fallback_reason == b.fallback_reason is None


def test_stage2_missing_file_keeps_p1(monkeypatch, tmp_path):
    """E13: ไม่มีไฟล์ β → p_final = p₁ + stage2_reason · ชั้น 1 และราคาไม่เปลี่ยน · shadow ไม่เปลี่ยนสิ่งที่ส่ง"""
    import forecast.live.stage2_live as S
    base = _run()
    monkeypatch.setattr(S, "STAGE2_DIR", tmp_path)
    for a, b in zip(_run(), base):
        assert a.p_final == a.p_stage1 == a.p_up == b.p_up
        assert "ไม่มีไฟล์ β" in a.stage2_reason
        assert a.price_quantiles == b.price_quantiles
