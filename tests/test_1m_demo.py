"""ชุดข้อมูลภาพ 1m สำหรับเดโม: หน้าต่างจบที่จุดตัด และป้ายมาจากราคาที่ใช้ให้คะแนน"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT, load_rules, session_times

RULES = load_rules()
TZ = RULES["timezone"]
DAYS = pd.bdate_range("2026-11-02", periods=3, tz=TZ)


@pytest.fixture(scope="module")
def demo():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("25_1m_scalogram_demo")


def bars():
    ends = []
    for d in DAYS:
        ends += list(pd.date_range(d + pd.Timedelta("10:01:00"), d + pd.Timedelta("12:30:00"), freq="1min"))
        ends += list(pd.date_range(d + pd.Timedelta("14:01:00"), d + pd.Timedelta("16:30:00"), freq="1min"))
    rng = np.random.default_rng(0)
    return pd.DataFrame({"bar_end": ends, "close": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(ends)))), "volume": rng.integers(100, 1000, len(ends)).astype(float)})


def labels(session, day):
    return pd.DataFrame({"symbol": ["KBANK"], "date": [day], "p_start": [100.0], "p_ref": [99.0], "p_end": [101.0], "direction": [1.0]})


@pytest.mark.parametrize("session", ["morning", "afternoon"])
def test_window_ends_at_or_before_the_submission_cutoff(demo, session):
    cutoff = session_times(RULES, session, DAYS[2])["cutoff"]
    win = demo.window_at_cutoff(bars(), cutoff, 80)
    assert len(win) == 80 and (win["bar_end"] <= cutoff).all()


@pytest.mark.parametrize("session", ["morning", "afternoon"])
def test_bars_after_cutoff_do_not_change_the_image(demo, session):
    b = bars()
    day = DAYS[2]
    cutoff = session_times(RULES, session, day)["cutoff"]
    changed = b.copy()
    after = changed["bar_end"] > cutoff
    assert after.any()
    changed.loc[after, "close"] *= 3.0
    changed.loc[after, "volume"] *= 50.0
    a = demo.build_dataset_for_symbol("KBANK", b, labels(session, day), RULES, session, 80)[0][0]["image"]
    c = demo.build_dataset_for_symbol("KBANK", changed, labels(session, day), RULES, session, 80)[0][0]["image"]
    assert np.array_equal(a, c)


def test_morning_window_uses_only_earlier_days(demo):
    cutoff = session_times(RULES, "morning", DAYS[2])["cutoff"]
    assert demo.window_at_cutoff(bars(), cutoff, 80)["bar_end"].max() < DAYS[2]


def test_prices_and_label_come_from_the_scored_row(demo):
    rows, _ = demo.build_dataset_for_symbol("KBANK", bars(), labels("afternoon", DAYS[2]), RULES, "afternoon", 80)
    r = rows[0]
    assert r["label"] == 1 and r["p_start"] == 100.0 and r["price_end"] == 99.0 and r["session_close"] == 101.0


@pytest.mark.parametrize("direction,expected", [(1.0, 1), (0.0, 0), (-1.0, 0)])
def test_flat_counts_as_not_up(demo, direction, expected):
    lab = labels("afternoon", DAYS[2]).assign(direction=direction)
    assert demo.build_dataset_for_symbol("KBANK", bars(), lab, RULES, "afternoon", 80)[0][0]["label"] == expected


def test_short_history_is_skipped_not_padded(demo):
    rows, skipped = demo.build_dataset_for_symbol("KBANK", bars().iloc[:10], labels("morning", DAYS[0]), RULES, "morning", 80)
    assert rows == [] and skipped == [str(DAYS[0].date())]


@pytest.fixture(scope="module")
def trainer():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("26_train_1m_cnn")


def test_walk_forward_plan_trains_only_on_earlier_days(trainer):
    plan = trainer.wf_plan(["d1", "d2", "d3", "d4"], 2)
    assert plan == [(["d1", "d2"], "d3"), (["d1", "d2", "d3"], "d4")]


def tiny_data(n_days=5, seed=0):
    rng = np.random.default_rng(seed)
    days = [f"2026-11-0{i + 1}" for i in range(n_days)]
    meta = pd.DataFrame([{"symbol": s, "date": d, "label": int(rng.integers(0, 2)), "price_end": 100.0 + rng.normal(), "session_close": 100.0 + rng.normal()} for d in days for s in ("KBANK", "ADVANC")])
    X = rng.random((len(meta), 3, 32, 32)).astype(np.float32)
    return X, meta["label"].to_numpy(dtype=np.float32), meta


def test_later_days_do_not_change_earlier_walk_forward_predictions(trainer):
    from types import SimpleNamespace
    import torch
    args = SimpleNamespace(epochs=1, batch_size=4, lr=1e-3, weight_decay=0.0)
    X, y, meta = tiny_data()
    dev = torch.device("cpu")
    a = trainer.walk_forward(X, y, meta, 3, args, dev, pretrained=False)
    X2, meta2 = X.copy(), meta.copy()
    last = (meta["date"] == meta["date"].max()).to_numpy()
    X2[last] = 0.0
    meta2.loc[last, "session_close"] = 999.0
    b = trainer.walk_forward(X2, y, meta2, 3, args, dev, pretrained=False)
    first = a["date"] == a["date"].min()
    assert np.allclose(a.loc[first, "pred_up_prob"], b.loc[first, "pred_up_prob"])
    assert np.allclose(a.loc[first, "pred_session_close"], b.loc[first, "pred_session_close"])


def test_summary_scores_model_against_simple_answers(trainer):
    p = pd.DataFrame({"date": ["a", "a"], "label": [1, 0], "pred_up_prob": [0.9, 0.2], "base_up_rate": [0.5, 0.5], "pred_session_close": [101.0, 100.0], "session_close": [100.0, 100.0], "price_end": [102.0, 100.0]})
    s = trainer.walk_forward_summary(p)
    assert s["accuracy_model_a"] == 1.0 and s["accuracy_always_not_up"] == 0.5
    assert np.isclose(s["rmse_model_b"], np.sqrt(0.5)) and np.isclose(s["rmse_reference_price"], np.sqrt(2.0))
