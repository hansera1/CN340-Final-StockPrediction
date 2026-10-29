import numpy as np
import pandas as pd
import pytest

from forecast.config import load_rules


def synthetic_bars(start_price, n_days, seed):
    rng = np.random.default_rng(seed)
    minutes = list(pd.timedelta_range("10:01:00", "12:30:00", freq="1min")) + \
        list(pd.timedelta_range("14:01:00", "16:30:00", freq="1min"))
    price, rows = start_price, []
    for d in pd.bdate_range("2026-01-05", periods=n_days):
        for m in minutes:
            o = price
            price *= np.exp(rng.normal(0, 0.001))
            rows.append((d + m, o, max(o, price), min(o, price), price, int(rng.integers(100, 1000))))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


@pytest.fixture
def rules():
    r = load_rules()
    r["bar_timestamp"] = "end"          # synthetic_bars ใช้เวลาจบแท่ง (10:01 … 16:30)
    r["evaluation"].update(min_train_days=20, test_block_days=5, holdout_days=5, n_boot=50)
    return r


@pytest.fixture
def raw_dir(tmp_path):
    for i, (sym, p) in enumerate({"KBANK": 150.0, "ADVANC": 300.0}.items()):
        synthetic_bars(p, 45, i).to_csv(tmp_path / f"{sym}_1min.csv", index=False)
    return tmp_path
