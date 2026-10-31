"""รอบที่ 4: fallback ข้อมูลรายวัน (ข้อ 3), P(c) รอบเช้าแบบราคาปิดทางการ (ข้อ 4), ราคาสอดคล้องทิศทาง (ข้อ 1ง)
ทดสอบกับข้อมูลจริงจะข้ามถ้าไม่มี data/raw"""
import copy

import numpy as np
import pandas as pd
import pytest

from forecast.config import RAW_DIR, load_rules, session_times, to_local
from forecast.features.dataset import assemble_dataset
from forecast.live.daily_check import prev_trading_day, synth_60m, synth_daily
from forecast.live.predict import consistent_point, official_close
from forecast.live.sources import live_bars

REAL = (RAW_DIR / "settrade" / "KBANK_1m.csv").exists() and (RAW_DIR / "yfinance" / "KBANK_60m.csv").exists()
DAYS = ["2026-10-27", "2026-10-28", "2026-10-29"]


def test_prev_trading_day_skips_weekend_and_holiday():
    hol = {pd.Timestamp("2026-10-23")}
    assert prev_trading_day("2026-11-02", hol) == pd.Timestamp("2026-10-30")      # จันทร์ → ศุกร์
    assert prev_trading_day("2026-10-26", hol) == pd.Timestamp("2026-10-22")      # ข้ามวันหยุดศุกร์ 23
    assert prev_trading_day("2026-10-28", hol) == pd.Timestamp("2026-10-27")


def test_consistent_point():
    assert consistent_point(250.0, 250.0, 0.60, 0.05) == 251.0
    assert consistent_point(250.0, 250.0, 0.40, 0.05) == 249.0
    assert consistent_point(250.0, 249.0, 0.52, 0.05) == 250.0    # กฎเลื่อนลงสวนทิศขึ้น → P(c)
    assert consistent_point(250.0, 251.0, 0.52, 0.05) == 251.0
    assert consistent_point(150.0, 150.0, 0.70, 0.05) == 150.5    # ช่องราคา 0.50 ช่วง 100–200


@pytest.mark.skipif(not REAL, reason="ไม่มีข้อมูลจริง")
@pytest.mark.parametrize("day", DAYS)
def test_synth_matches_real(day):
    rules, d = load_rules(), pd.Timestamp(day)
    for sym in rules["symbols"]:
        row, how = synth_daily(sym, rules, RAW_DIR, d)
        assert "Settrade 1m" in how
        real = pd.read_csv(RAW_DIR / "settrade" / f"{sym}_1d.csv", parse_dates=["timestamp"])
        real = real[real["timestamp"] == d].iloc[0]
        for c in ["open", "high", "low", "close"]:
            assert row[c].iat[0] == real[c]
        h, _ = synth_60m(sym, rules, RAW_DIR, d)
        y = pd.read_csv(RAW_DIR / "yfinance" / f"{sym}_60m.csv", parse_dates=["timestamp"])
        y = y[y["timestamp"].dt.normalize() == d].set_index("timestamp")
        assert np.allclose(h.set_index("timestamp")["close"], y["close"].reindex(h["timestamp"]))


def _live_row(rules, day, sess):
    date = to_local(day, rules).normalize()
    as_of = session_times(rules, sess, date)["cutoff"]
    by = {}
    for sym in rules["symbols"]:
        b, _ = live_bars(sym, rules, date, as_of)
        by[sym] = b[b["bar_end"] <= as_of].reset_index(drop=True)
    ds = assemble_dataset(rules, bars_by_symbol=by, as_of=as_of, live_row=(date, sess))
    return ds[(ds["date"] == date) & (ds["session"] == sess)].set_index("symbol")


@pytest.mark.skipif(not REAL, reason="ไม่มีข้อมูลจริง")
@pytest.mark.parametrize("pref", ["last_bar_close", "official_close"])
def test_live_price_features_match_backtest(pref):
    """แถววันจริงต้องมี P(c) และ fl_* เท่ากับ backtest (เดิม p_ref ของแถววันจริงเป็น NaN → fl_exp_ticks หาย)
    และ official_close: P(c) รอบเช้า = ราคาปิดทางการที่ median_rule ของวันจริงใช้"""
    rules = copy.deepcopy(load_rules())
    rules["label"]["p_ref_morning"] = pref
    end = to_local(DAYS[-1], rules) + pd.Timedelta(hours=23, minutes=59)
    bt = assemble_dataset(rules, as_of=end)
    cols = ["p_ref", "fl_exp_ticks", "fl_prev_range_ticks", "fl_flat20_price", "fl_rv_intra"]
    for day in DAYS:
        for sess in ["morning", "afternoon"]:
            live = _live_row(rules, day, sess)
            b = bt[(bt["date"] == to_local(day, rules)) & (bt["session"] == sess)].set_index("symbol")
            for sym in rules["symbols"]:
                assert np.allclose(live.loc[sym, cols].astype(float), b.loc[sym, cols].astype(float)), (day, sess, sym)
                if pref == "official_close" and sess == "morning":
                    p, _, _ = official_close(sym, to_local(day, rules), rules)
                    assert live.loc[sym, "p_ref"] == p
