"""ข้อมูลหลังจุดตัดต้องไม่มีผลต่อ features ของรอบนั้น (หัวข้อ 02, 04)"""
import numpy as np
import pandas as pd
import pytest

from forecast.config import session_times
from forecast.features.dataset import assemble_dataset, feature_cols


@pytest.mark.parametrize("session", ["morning", "afternoon"])
@pytest.mark.parametrize("spectral", [False, True])
def test_future_data_does_not_change_features(rules, raw_dir, session, spectral):
    base = assemble_dataset(rules, raw_dir=raw_dir, spectral=spectral, ext_dir=raw_dir / "none")
    date = base["date"].drop_duplicates().iloc[30]
    cutoff = session_times(rules, session, date)["cutoff"]

    # บิดเบือนทุกแท่งหลังจุดตัด (ทั้งสองหุ้น)
    for sym in rules["symbols"]:
        path = raw_dir / f"{sym}_1m.csv"
        df = pd.read_csv(path)
        ts = pd.to_datetime(df["timestamp"]).dt.tz_localize(rules["timezone"])
        after = ts > cutoff
        df.loc[after, ["open", "high", "low", "close"]] *= 1.5
        df.loc[after, "volume"] *= 7
        df.to_csv(path, index=False)

    moved = assemble_dataset(rules, raw_dir=raw_dir, spectral=spectral, ext_dir=raw_dir / "none")
    key = (base["date"] == date) & (base["session"] == session)
    cols = feature_cols(base)
    a = base.loc[key, cols].reset_index(drop=True)
    b = moved.loc[(moved["date"] == date) & (moved["session"] == session), cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # และ label ต้องเปลี่ยนจริง (ยืนยันว่าทดสอบมีความหมาย)
    assert not np.allclose(base.loc[key, "p_end"], moved.loc[key, "p_end"])


def test_live_row_matches_backtest_row(rules, raw_dir):
    """แถวที่สร้างแบบวันจริง (ตัดข้อมูลที่ as_of) ต้องได้ features เท่ากับแถวเดียวกันใน backtest"""
    full = assemble_dataset(rules, raw_dir=raw_dir, ext_dir=raw_dir / "none")
    date = full["date"].drop_duplicates().iloc[-1]
    for session in ("morning", "afternoon"):
        cutoff = session_times(rules, session, date)["cutoff"]
        live = assemble_dataset(rules, raw_dir=raw_dir, ext_dir=raw_dir / "none",
                                as_of=cutoff, live_row=(date, session))
        cols = feature_cols(full)
        a = full[(full["date"] == date) & (full["session"] == session)][cols].reset_index(drop=True)
        b = live[(live["date"] == date) & (live["session"] == session)][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
