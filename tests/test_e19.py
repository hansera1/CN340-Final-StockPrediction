"""E19: ภาพ 1m ไม่รั่วข้อมูล และส่วนประกอบใช้ข้อมูลฝึกเท่านั้น"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT, load_rules
from forecast.features.images import morlet_cwt
from forecast.features.scalogram_1m import MIN_BARS, image_for_day, pick_source

RULES = load_rules()
TZ = RULES["timezone"]
DAY = pd.Timestamp("2026-11-06", tz=TZ)


def bars(last_am="12:29", gap=None):
    am = pd.date_range(DAY + pd.Timedelta("10:00:00"), DAY + pd.Timedelta(f"{last_am}:00"), freq="1min")
    pm = pd.date_range(DAY + pd.Timedelta("14:00:00"), DAY + pd.Timedelta("16:29:00"), freq="1min")
    t = am.append(pm)
    rng = np.random.default_rng(0)
    b = pd.DataFrame({"bar_start": t, "bar_end": t + pd.Timedelta(minutes=1), "date": DAY, "close": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(t)))),
                      "volume": rng.integers(100, 1000, len(t)).astype(float)})
    return b if gap is None else b.drop(index=gap).reset_index(drop=True)


def test_bars_after_morning_close_have_no_effect():
    b = bars()
    base = image_for_day(b, DAY, RULES)
    moved = b.copy()
    after = moved["bar_end"] > DAY + pd.Timedelta("12:30:00")
    moved.loc[after, "close"] *= 1.5
    moved.loc[after, "volume"] *= 9
    np.testing.assert_array_equal(base, image_for_day(moved, DAY, RULES))
    assert base.shape == (3, 224, 224) and base.dtype == np.uint8


def test_last_morning_bar_is_used():
    b = bars()
    moved = b.copy()
    moved.loc[moved["bar_end"] == DAY + pd.Timedelta("12:30:00"), "close"] *= 1.1
    assert not np.array_equal(image_for_day(b, DAY, RULES), image_for_day(moved, DAY, RULES))


def test_too_few_bars_gives_no_image_and_falls_back_to_other_source():
    few = bars(last_am="11:30")
    assert len(few[few["bar_end"] <= DAY + pd.Timedelta("12:30:00")]) < MIN_BARS
    assert image_for_day(few, DAY, RULES) is None
    assert pick_source(few, bars(), DAY, RULES)[1] == "yfinance"
    assert pick_source(bars(), few, DAY, RULES)[1] == "settrade"
    assert pick_source(few, None, DAY, RULES) == (None, None)


def test_missing_minutes_still_build_one_image():
    assert image_for_day(bars(gap=list(range(10, 20))), DAY, RULES).shape == (3, 224, 224)


def test_morlet_default_scales_unchanged_and_s_min_starts_at_nyquist():
    x = np.sin(np.arange(150) / 3.0)
    np.testing.assert_array_equal(morlet_cwt(x, 32), morlet_cwt(x, 32, s_min=1.0))
    assert not np.array_equal(morlet_cwt(x, 32), morlet_cwt(x, 32, s_min=2.0))


@pytest.fixture(scope="module")
def e19():
    spec = importlib.util.spec_from_file_location("e19_train", ROOT / "experiments" / "28_e19_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pca_is_fit_on_training_rows_only(e19):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 20))
    tr = np.arange(30) < 20
    a = e19.pca_features(emb, tr, 3)
    moved = emb.copy()
    moved[~tr] *= 100
    np.testing.assert_allclose(a[tr], e19.pca_features(moved, tr, 3)[tr])


def test_oof_folds_keep_each_day_together_and_use_only_training_days(e19):
    d = np.repeat(np.arange(10), 2)
    tr = d < 8
    folds = e19.oof_days(d, tr)
    assert np.logical_or.reduce(folds).sum() == tr.sum()
    for f in folds:
        assert not (f & ~tr).any()
        assert not set(d[f]) & set(d[tr & ~f])
