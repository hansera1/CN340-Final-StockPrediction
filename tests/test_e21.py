"""E21: ลำดับข้อมูลไม่รั่วข้อมูลหลังจุดตัด และโครงข่ายรับ input ได้ครบ"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT, load_rules, session_times

RULES = load_rules()
TZ = RULES["timezone"]


@pytest.fixture(scope="module")
def e21():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("44_e21_ablation")


def bars(n_days=40, seed=0):
    days = pd.bdate_range("2026-01-05", periods=n_days, tz=TZ)
    ends = [d + pd.Timedelta(hours=h) for d in days for h in (11, 12, 13, 15, 16, 17)]
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, len(ends))))
    return pd.DataFrame({"bar_end": ends, "date": [e.normalize() for e in ends], "open": c * 0.999, "high": c * 1.002, "low": c * 0.998, "close": c, "volume": rng.integers(100, 1000, len(ends)).astype(float)})


@pytest.mark.parametrize("session", ["morning", "afternoon"])
def test_bars_after_cutoff_have_no_effect(e21, session):
    b = bars()
    day = b["date"].unique()[30]
    rows = pd.DataFrame({"date": [day], "session": [session]})
    cutoff = session_times(RULES, session, day)["cutoff"]
    base = e21.build_seqs(b, rows, RULES)
    changed = b.copy()
    after = changed["bar_end"] > cutoff
    assert after.any()
    changed.loc[after, ["open", "high", "low", "close"]] *= 3.0
    changed.loc[after, "volume"] *= 50.0
    assert np.array_equal(base, e21.build_seqs(changed, rows, RULES))


def test_volume_surprise_uses_only_earlier_days(e21):
    b = bars()
    base = e21.add_channels(b)
    changed = b.copy()
    changed.loc[len(b) - 1, "volume"] *= 100.0
    new = e21.add_channels(changed)
    assert np.allclose(base["c_vs"].to_numpy()[:-1], new["c_vs"].to_numpy()[:-1])
    assert base["c_vs"].iloc[-1] != new["c_vs"].iloc[-1]


def test_same_time_of_day_is_the_volume_baseline(e21):
    b = bars()
    b["volume"] = np.where(pd.DatetimeIndex(b["bar_end"]).hour == 11, 10000.0, 100.0)
    c = e21.add_channels(b)
    assert np.allclose(c["c_vs"].to_numpy()[-12:], 0.0)


def test_return_features_shape_and_values(e21):
    seq = np.zeros((1, len(e21.CHANNELS), e21.LENGTH), dtype=np.float32)
    seq[0, 0, -1], seq[0, 0, -6], seq[0, 0, -28] = 1.0, 2.0, 4.0
    assert np.allclose(e21.return_features(seq)[0], [1.0, 3.0, 7.0])


@pytest.mark.parametrize("kind,shape,n_tab", [("cnn1d_3", (4, 3, 56), 0), ("cnn1d_ext", (4, 8, 56), 0), ("cnn2d_small", (4, 3, 56, 56), 0), ("fusion", (4, 8, 56), 5)])
def test_nets_output_one_logit_per_row(e21, kind, shape, n_tab):
    torch = pytest.importorskip("torch")
    net = e21.make_net(kind, n_tab)
    out = net(torch.zeros(shape), torch.zeros(4, 1), torch.zeros(4, max(n_tab, 1)))
    assert out.shape == (4,)


def test_prediction_uses_only_training_rows_for_scaling(e21):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=(n, 8, 56)).astype(np.float32)
    y = rng.integers(0, 2, n)
    dates = np.repeat(np.arange(30), 2)
    tr, te = dates < 25, dates >= 25
    import torch
    dev = torch.device("cpu")
    a = e21.fit_predict_net("cnn1d_ext", x, np.zeros(n), None, y, tr, te, dates, dev, (0,), 1, "f", [])
    changed = x.copy()
    changed[te] += 100.0
    tr_only_same = e21.fit_predict_net("cnn1d_ext", changed, np.zeros(n), None, y, tr, tr, dates, dev, (0,), 1, "f", [])
    base = e21.fit_predict_net("cnn1d_ext", x, np.zeros(n), None, y, tr, tr, dates, dev, (0,), 1, "f", [])
    assert a.shape == (te.sum(),)
    assert np.allclose(tr_only_same, base, atol=1e-5)
