"""รวมแท่ง 1m → 60m แบบ yfinance และความสอดคล้องระหว่าง backtest (ไฟล์ 60m) กับวันจริง (1m วันนี้)"""
import pandas as pd
import pytest
from conftest import synthetic_bars

from forecast.config import session_times
from forecast.data.io import OHLCV, aggregate_bars, frame_from_raw, load_bars
from forecast.features.dataset import assemble_dataset, feature_cols
from forecast.live.sources import live_bars


@pytest.fixture
def sources(rules, tmp_path):
    """yfinance/: <SYM>_1m.csv + <SYM>_60m.csv (60m สร้างจาก 1m ด้วย aggregate_bars); settrade/: ว่าง"""
    r = {**rules, "bar_timestamp": "start", "data": {"source": "yfinance", "interval": "60m"},
         "live": {**rules["live"], "sources": ["settrade", "yfinance"], "max_bar_lag_min": 5}}
    dirs = {s: tmp_path / s for s in ("settrade", "yfinance")}
    for d in dirs.values():
        d.mkdir()
    for i, (sym, p) in enumerate({"KBANK": 150.0, "ADVANC": 300.0}.items()):
        m1 = synthetic_bars(p, 45, i)
        m1["timestamp"] -= pd.Timedelta(minutes=1)            # เป็นเวลาเริ่มแท่งแบบ yfinance / Settrade
        m1.to_csv(dirs["yfinance"] / f"{sym}_1m.csv", index=False)
        h = aggregate_bars(frame_from_raw(m1, r, "1m"), 60, r)
        h["timestamp"] = h["bar_start"].dt.tz_localize(None)
        h[["timestamp"] + OHLCV].to_csv(dirs["yfinance"] / f"{sym}_60m.csv", index=False)
    return r, dirs, tmp_path / "none"


def test_aggregate_like_yfinance(rules):
    r = {**rules, "bar_timestamp": "start"}
    d = pd.Timestamp("2026-10-29")
    mins = ["09:57", "09:59", "10:00", "10:59", "12:29", "13:58", "14:00", "16:29", "16:35"]
    raw = pd.DataFrame({"timestamp": [d + pd.Timedelta(f"{m}:00") for m in mins],
                        "open": range(9), "high": range(9), "low": range(9),
                        "close": [float(x) for x in range(9)], "volume": [10] * 9})
    agg = aggregate_bars(frame_from_raw(raw, r, "1m"), 60, r)
    got = {b.strftime("%H:%M"): e.strftime("%H:%M") for b, e in zip(agg["bar_start"], agg["bar_end"])}
    # ไม่มีแท่งก่อน 10:00 และแท่งประมูล 16:35; 13:58 อยู่ในแท่ง 13:00; แท่งคร่อมปิดภาคจบที่ 12:30 / 16:30
    assert got == {"10:00": "11:00", "12:00": "12:30", "13:00": "14:00", "14:00": "15:00", "16:00": "16:30"}
    first = agg.iloc[0]
    assert (first["open"], first["close"], first["volume"]) == (2, 3.0, 20)


def test_afternoon_features_ignore_unfinished_12h_bar(sources):
    """แท่ง 60m ที่เริ่ม 12:00 (จบ 12:30) ยังไม่จบ ณ จุดตัด 12:15 → แก้ค่าแล้ว features รอบบ่ายต้องไม่เปลี่ยน"""
    r, dirs, none = sources
    base = assemble_dataset(r, raw_dir=dirs["yfinance"], ext_dir=none)
    date = base["date"].drop_duplicates().iloc[30]
    for sym in r["symbols"]:
        path = dirs["yfinance"] / f"{sym}_60m.csv"
        df = pd.read_csv(path, parse_dates=["timestamp"])
        hit = df["timestamp"] == date.tz_localize(None) + pd.Timedelta("12:00:00")
        df.loc[hit, ["open", "high", "low", "close"]] *= 1.5
        df.loc[hit, "volume"] *= 9
        df.to_csv(path, index=False)
    moved = assemble_dataset(r, raw_dir=dirs["yfinance"], ext_dir=none)
    key = lambda d: (d["date"] == date) & (d["session"] == "afternoon")
    cols = feature_cols(base)
    pd.testing.assert_frame_equal(base.loc[key(base), cols].reset_index(drop=True),
                                  moved.loc[key(moved), cols].reset_index(drop=True))
    m = lambda d: (d["date"] == date) & (d["session"] == "morning")
    assert (base.loc[m(base), "p_end"].to_numpy() != moved.loc[m(moved), "p_end"].to_numpy()).all()


@pytest.mark.parametrize("session", ["morning", "afternoon"])
def test_live_from_1m_matches_backtest_60m(sources, session):
    """วันจริง: ประวัติ 60m + 1m วันนี้ (รวมเป็น 60m) → features เท่ากับแถวเดียวกันใน backtest จากไฟล์ 60m"""
    r, dirs, none = sources
    full = assemble_dataset(r, raw_dir=dirs["yfinance"], ext_dir=none)
    date = full["date"].drop_duplicates().iloc[-1]
    cutoff = session_times(r, session, date)["cutoff"]
    by, metas = {}, {}
    for sym in r["symbols"]:
        b, metas[sym] = live_bars(sym, r, date, cutoff, dirs["yfinance"], dirs)
        by[sym] = b[b["bar_end"] <= cutoff].reset_index(drop=True)
    live = assemble_dataset(r, bars_by_symbol=by, as_of=cutoff, live_row=(date, session), ext_dir=none)
    cols = feature_cols(full)
    a = full[(full["date"] == date) & (full["session"] == session)][cols].reset_index(drop=True)
    b = live[(live["date"] == date) & (live["session"] == session)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    expected = "history-only" if session == "morning" else "yfinance"   # settrade ไม่มีไฟล์ → สำรอง
    assert all(m["source"] == expected for m in metas.values())


def test_stale_source_falls_back(sources):
    r, dirs, _ = sources
    date = load_bars("KBANK", r, dirs["yfinance"], "1m")["date"].max()
    for sym in r["symbols"]:   # Settrade ค้างที่ 10:30 → เก่ากว่า 5 นาที ณ 12:15
        df = pd.read_csv(dirs["yfinance"] / f"{sym}_1m.csv", parse_dates=["timestamp"])
        df[df["timestamp"] < date.tz_localize(None) + pd.Timedelta("10:30:00")].to_csv(
            dirs["settrade"] / f"{sym}_1m.csv", index=False)
    cutoff = session_times(r, "afternoon", date)["cutoff"]
    _, meta = live_bars("KBANK", r, date, cutoff, dirs["yfinance"], dirs)
    assert meta["source"] == "yfinance" and meta["lag_sec"] == 0
    assert "settrade: แท่งล่าสุดเก่า" in meta["skipped"][0]
