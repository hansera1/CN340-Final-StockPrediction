"""การประเมินทางการเงิน: XIRR, Sharpe, ค่าธรรมเนียม และ drawdown ตรงกับการคำนวณด้วยมือ"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT


@pytest.fixture(scope="module")
def fin():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("61_financial_eval")


def test_xirr_matches_one_year_hand_example(fin):
    assert np.isclose(fin.xirr(["2025-01-01", "2026-01-01"], [-1000, 1100]), 0.10, atol=1e-9)


def test_xirr_two_year_hand_example(fin):
    # −1000 วันแรก, +1210 หลัง 730 วัน → (1.21)^(365/730) − 1 = 10%
    assert np.isclose(fin.xirr(["2025-01-01", "2027-01-01"], [-1000, 1210]), 0.10, atol=1e-9)


def test_no_trades_gives_zero_return(fin):
    assert fin.xirr(["2025-01-01", "2025-01-02", "2025-01-02"], [-100, 0, 100]) == 0.0


def test_fee_applies_on_both_legs(fin):
    pnl = fin.trade_pnl([0.01], [1], 0.001)[0]
    assert np.isclose(pnl, 1000 - 0.001 * 100_000 * 2.01)
    assert fin.trade_pnl([0.05], [0], 0.001)[0] == 0.0


def test_short_gains_when_price_falls(fin):
    assert fin.trade_pnl([-0.02], [-1], 0.0)[0] == pytest.approx(2000.0)


def test_sharpe_formula(fin):
    d = np.array([0.01, -0.005, 0.002, 0.004])
    assert np.isclose(fin.sharpe(d, 0.0), d.mean() / d.std(ddof=1) * np.sqrt(252))


def test_max_drawdown(fin):
    assert np.isclose(fin.max_drawdown([100, 120, 90, 130]), -0.25)


def test_signal_thresholds_by_session(fin):
    keys = {"symbol": ["KBANK"], "session": ["morning"], "date": [pd.Timestamp("2026-01-05", tz="Asia/Bangkok")]}
    ds = pd.DataFrame({**keys, "p_start": [100.0], "p_end": [101.0], "direction": [1.0]})
    am = fin.signal_rows(ds, pd.DataFrame({**keys, "p": [0.5]}), pd.DataFrame(columns=["date", "symbol", "session", "p"]))
    assert bool(am["up"].iloc[0])
    pm_keys = {**keys, "session": ["afternoon"]}
    ds2 = pd.DataFrame({**pm_keys, "p_start": [100.0], "p_end": [101.0], "direction": [1.0]})
    pm = fin.signal_rows(ds2, pd.DataFrame(columns=["date", "symbol", "session", "p"]), pd.DataFrame({**pm_keys, "p": [0.5]}))
    assert not bool(pm["up"].iloc[0])
