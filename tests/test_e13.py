"""E13 E-5: ป้องกันข้อมูลรั่วของชั้น 2 (results/e13_preregistration.md)"""
import numpy as np
import pandas as pd
import pytest

from forecast.config import OUTPUT_DIR, RAW_DIR, load_rules, raw_source_dir, session_times, to_local
from forecast.data.io import aggregate_bars, load_bars
from forecast.features.fine import F5, f5_features
from forecast.models.stage2 import Stage2, fit_for_day, training_rows

RULES = load_rules()
DAY = to_local("2026-10-28", RULES).normalize()


def _bars5(day=DAY, seed=0):
    rng = np.random.default_rng(seed)
    start = day + pd.Timedelta(hours=10)
    t = pd.date_range(start, day + pd.Timedelta(hours=12, minutes=25), freq="5min")
    c = 250 + np.cumsum(rng.choice([-1.0, 0.0, 1.0], len(t)))
    return pd.DataFrame({"bar_start": t, "bar_end": t + pd.Timedelta(minutes=5), "open": c, "high": c + 1,
                         "low": c - 1, "close": c, "date": day})


def test_f5_ignores_bars_after_cutoff_and_uses_bars_before():
    b = _bars5()
    cut = session_times(RULES, "afternoon", DAY)["cutoff"]
    base = f5_features(b, DAY, RULES)
    after = b.copy()
    after.loc[after["bar_end"] > cut, ["open", "high", "low", "close"]] *= 1.5     # แท่งหลังจุดตัดเปลี่ยน
    assert f5_features(after, DAY, RULES) == base
    before = b.copy()
    before.loc[before["bar_end"] == before.loc[before["bar_end"] <= cut, "bar_end"].max(), "close"] += 3
    assert f5_features(before, DAY, RULES) != base                                # แท่งสุดท้ายก่อนจุดตัดมีผล


def test_f5_shift_cutoff_changes_only_by_removed_bars():
    """เลื่อนจุดตัด 5 นาที = ใช้แท่งที่จบ ≤ จุดตัดใหม่เท่านั้น: เท่ากับคำนวณจากชุดที่ตัดแท่งนั้นออกเองแล้วใช้จุดตัดใหม่"""
    import copy
    b = _bars5(seed=3)
    r2 = copy.deepcopy(RULES)
    h, m = map(int, RULES["sessions"]["afternoon"]["cutoff"].split(":"))
    r2["sessions"]["afternoon"]["cutoff"] = f"{h:02d}:{m - 5:02d}"
    cut2 = session_times(r2, "afternoon", DAY)["cutoff"]
    assert f5_features(b, DAY, r2) == f5_features(b[b["bar_end"] <= cut2], DAY, r2)


def _table(n_days=30, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-09-01", periods=n_days, tz=RULES["timezone"])
    rows = []
    for d in days:
        for sym in ["KBANK", "ADVANC"]:
            rows.append({"date": d, "symbol": sym, "p1": rng.uniform(0.4, 0.6), "direction": rng.choice([-1, 0, 1, 1]),
                         **{k: rng.normal() for k in F5}})
    return pd.DataFrame(rows)


def test_training_rows_only_before_date():
    t = _table()
    date = t["date"].unique()[25]
    tr, days = training_rows(t, date, 20)
    assert len(days) == 20 and max(days) < date and (tr["date"] < date).all()
    assert (tr["direction"] != 0).all()


def test_outlier_on_test_day_does_not_change_scaler():
    t = _table()
    date = t["date"].unique()[25]
    m1, _ = fit_for_day(t, date)
    t2 = t.copy()
    t2.loc[t2["date"] == date, F5] = 1e6                       # วันทดสอบผิดปกติ
    t2.loc[t2["date"] > date, F5] = -1e6                       # และวันหลังจากนั้น
    m2, _ = fit_for_day(t2, date)
    assert np.allclose(m1.mean_, m2.mean_) and np.allclose(m1.sd_, m2.sd_) and np.allclose(m1.beta_, m2.beta_)


def test_stage2_shrinks_and_offsets():
    """λ = 45: β เล็ก · x = mean → p_final = p₁ (ไม่มี intercept)"""
    t = _table()
    m, info = fit_for_day(t, t["date"].unique()[25])
    assert info["reason"] is None and np.all(np.abs(m.beta_) < 0.5)
    p = m.predict(m.mean_[None, :], [0.53])
    assert p[0] == pytest.approx(0.53)
    assert Stage2.from_dict(m.to_dict()).predict(m.mean_[None, :] + 1, [0.5]) == pytest.approx(m.predict(m.mean_[None, :] + 1, [0.5]))


def test_too_few_nonflat_returns_none():
    t = _table(n_days=22)
    t["direction"] = 0
    t.loc[t.index[:10], "direction"] = 1
    m, info = fit_for_day(t, t["date"].unique()[21])
    assert m is None and "p_final = p₁" in info["reason"]


REAL = (RAW_DIR / "settrade" / "KBANK_1m.csv").exists() and (RAW_DIR / "yfinance" / "KBANK_5m.csv").exists()


@pytest.mark.skipif(not REAL, reason="ไม่มีข้อมูลจริง")
def test_live_path_f5_equals_backtest():
    """วันที่มีทั้งสองแหล่ง: Settrade 1m → 5m (เส้นทางวันจริง) = yfinance 5m (backtest) ทุกค่า"""
    for sym in RULES["symbols"]:
        st = load_bars(sym, RULES, raw_source_dir(RULES, "settrade"), "1m")
        yf = load_bars(sym, RULES, raw_source_dir(RULES, "yfinance"), "5m")
        days = sorted(set(st["date"]) & set(yf["date"]))
        assert days
        for d in days:
            a = f5_features(aggregate_bars(st[st["date"] == d], 5, RULES), d, RULES)
            b = f5_features(yf[yf["date"] == d], d, RULES)
            assert np.allclose([a[k] for k in F5], [b[k] for k in F5]), (sym, d)


@pytest.mark.skipif(not (OUTPUT_DIR / "experiments" / "E12" / "fd_lr_predictions.csv").exists(), reason="ไม่มีผล E12")
def test_p1_out_of_sample():
    """p₁ ทุกแถว: walk-forward มาจาก fold ที่วันฝึกทั้งหมด < วันที่ทาย · holdout ฝึกก่อนวันแรกของ holdout"""
    from forecast.evaluation.runner import load_or_build_dataset
    from forecast.evaluation.splits import holdout_split, walk_forward_splits
    ev = RULES["evaluation"]
    ds = load_or_build_dataset({**RULES, "label": {**RULES["label"], "p_ref_morning": "last_bar_close"}})
    pm = ds[ds["session"] == "afternoon"]
    dates = pm["date"].to_numpy()
    tz = RULES["timezone"]
    wf = pd.read_csv(OUTPUT_DIR / "experiments" / "E12" / "fd_lr_predictions.csv")
    wf = pd.to_datetime(wf.loc[wf["session"] == "afternoon", "date"], utc=True).dt.tz_convert(tz)
    trained_before = {}
    for tr, te in walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]):
        for d in te:
            trained_before[pd.Timestamp(d)] = pd.Timestamp(max(tr))
    assert all(trained_before[d] < d for d in wf)
    ho = pd.read_csv(OUTPUT_DIR / "experiments" / "E12_holdout" / "fd_lr_predictions.csv")
    ho = pd.to_datetime(ho["date"], utc=True).dt.tz_convert(tz)
    tr, te = next(iter(holdout_split(dates, ev["holdout_days"])))
    assert pd.Timestamp(max(tr)) < ho.min()
