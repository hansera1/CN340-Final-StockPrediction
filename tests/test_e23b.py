"""E23b: จับคู่ภาพ 1m ตรงแถว และคะแนนคิดเฉพาะแบบที่มีผล"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT


@pytest.fixture(scope="module")
def e23b():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("49_e23b_1m")


def rows():
    days = pd.to_datetime(["2026-11-04", "2026-11-04", "2026-11-05"]).tz_localize("Asia/Bangkok")
    return pd.DataFrame({"date": days, "symbol": ["KBANK", "ADVANC", "KBANK"], "session": "morning", "p_ref": [100.0, 200.0, 101.0], "p_end": [101.0, 201.0, 102.0]})


def test_images_follow_their_symbol_and_day_not_file_order(e23b):
    meta = pd.DataFrame({"symbol": ["KBANK", "ADVANC", "KBANK"], "date": ["2026-11-05", "2026-11-04", "2026-11-04"], "price_end": [101.0, 200.0, 100.0], "session_close": [102.0, 201.0, 101.0]})
    X = np.array([5, 6, 7]).reshape(3, 1)
    g, im = e23b.join_images(rows(), meta, X)
    got = dict(zip(zip(g["symbol"], g["date"].astype(str).str[:10]), im[:, 0]))
    assert got[("KBANK", "2026-11-05")] == 5 and got[("ADVANC", "2026-11-04")] == 6 and got[("KBANK", "2026-11-04")] == 7


def test_price_mismatch_stops_the_run(e23b):
    meta = pd.DataFrame({"symbol": ["KBANK", "ADVANC", "KBANK"], "date": ["2026-11-05", "2026-11-04", "2026-11-04"], "price_end": [101.0, 200.0, 999.0], "session_close": [102.0, 201.0, 101.0]})
    with pytest.raises(AssertionError):
        e23b.join_images(rows(), meta, np.zeros((3, 1)))


def test_score_reports_rmse_mae_and_shift(e23b):
    x = pd.DataFrame({"session": ["morning", "morning"], "symbol": ["KBANK", "ADVANC"], "p_ref": [100.0, 200.0], "p_end": [103.0, 200.0], "pt_m": [100.0, 201.0]})
    t = e23b.score(x, ["m"]).set_index("group")
    assert np.isclose(t.loc["เช้า", "rmse"], np.sqrt(5.0)) and t.loc["เช้า", "mae"] == 2.0 and t.loc["เช้า", "mean_abs_shift"] == 0.5
