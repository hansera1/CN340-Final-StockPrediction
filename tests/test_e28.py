"""E28: pretrain / fine-tune / test ไม่ซ้อนกัน และไม่มีข้อมูลรั่วระหว่างช่วง"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT

pytest.importorskip("scipy")
TZ = "Asia/Bangkok"
SHARED = ["x1", "x2", "x3"]


@pytest.fixture(scope="module")
def e28():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("56_e28_disjoint_finetune")


def frame(n_days=260, pm_from=30, seed=0):
    """สองหุ้น สองรอบ · รอบบ่ายเริ่มวันที่ pm_from · ป้ายขึ้นกับ x1"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days, tz=TZ)
    rows = []
    for i, day in enumerate(days):
        for sym in ("KBANK", "ADVANC"):
            for sess in ("morning", "afternoon"):
                if sess == "afternoon" and i < pm_from:
                    continue
                x = rng.normal(size=3)
                up = rng.random() < 1 / (1 + np.exp(-(0.8 * x[0] - 0.2)))
                rows.append({"date": day, "symbol": sym, "session": sess, "x1": x[0], "x2": x[1], "x3": x[2],
                             "f_gap_open": rng.normal(), "f_rv_intra": abs(rng.normal()),
                             "d_sess_pm": float(sess == "afternoon"), "d_sym_advanc": float(sym == "ADVANC"),
                             "direction": 1.0 if up else rng.choice([0.0, -1.0])})
    return pd.DataFrame(rows)


def test_periods_are_ordered_and_disjoint(e28):
    d = frame()
    days = np.sort(d["date"].unique())
    for start in days[150::20]:
        pre, ft = e28.split_days(d["date"].to_numpy(), start)
        assert len(pre) and len(ft)
        assert pre.max() < ft.min() and ft.max() < start
        assert not set(pre) & set(ft)


def test_gap_is_five_trading_days(e28):
    d = frame()
    labeled = d["date"].to_numpy()
    ud = np.unique(labeled)
    start = ud[200]
    pre, ft = e28.split_days(labeled, start)
    between = ud[(ud > pre.max()) & (ud < ft.min())]
    assert len(between) == 5
    assert len(pre) + 5 + len(ft) == 200


def fit_block(e28, d, start, session="afternoon", kind="D-prime"):
    tr = d[d["date"] < start]
    pre_days, ft_days = e28.split_days(tr["date"].to_numpy(), start)
    pre, ft = tr[tr["date"].isin(pre_days)], tr[(tr["session"] == session) & tr["date"].isin(ft_days)]
    pre_cols, ft_cols, const = e28.session_cols(SHARED, session)
    m = e28.DisjointFineTune()
    if kind == "Pre70":
        m.ft_days_min = 10 ** 9
    return m.fit(e28.s31.part(pre), e28.s31.part(ft), pre_cols, ft_cols, const), pre, ft


def test_changing_fine_tune_rows_leaves_pretrain_unchanged(e28):
    d = frame()
    start = np.sort(d["date"].unique())[220]
    a, _, ft = fit_block(e28, d, start)
    changed = d.copy()
    rows = changed.index.isin(ft.index)
    changed.loc[rows, SHARED] *= 50.0
    changed.loc[rows, "direction"] = 1.0
    b, _, _ = fit_block(e28, changed, start)
    assert np.allclose(a.scaler.mean, b.scaler.mean) and np.allclose(a.scaler.std, b.scaler.std)
    assert np.allclose(a.beta_pre, b.beta_pre) and np.isclose(a.a_pre, b.a_pre) and a.C == b.C


def test_test_block_and_future_labels_do_not_change_predictions(e28):
    d = frame()
    days = np.sort(d["date"].unique())
    test = days[220:225]
    a, _ = e28.block(d, SHARED, "morning", test)
    changed = d.copy()
    later = changed["date"] >= test.min()
    changed.loc[later, "direction"] = 1.0
    changed.loc[changed["date"] > test.max(), SHARED] = 99.0
    b, _ = e28.block(changed, SHARED, "morning", test)
    assert np.allclose(a["p_up"].to_numpy(), b["p_up"].to_numpy())


def test_infinite_kappa_equals_pre70(e28):
    d = frame(n_days=180)
    start = np.sort(d["date"].unique())[120]     # ft ราว 36 วัน < 40 → κ = ∞
    m, _, ft = fit_block(e28, d, start, "morning")
    assert ft["date"].nunique() < 40 and not np.isfinite(m.kappa)
    p, _, _ = fit_block(e28, d, start, "morning", "Pre70")
    x = d[d["date"] >= start].head(20)
    assert np.allclose(m.predict(x), p.predict(x))


def test_morning_has_no_afternoon_features_and_session_dummy_goes_to_intercept(e28):
    d = frame()
    start = np.sort(d["date"].unique())[220]
    m, _, _ = fit_block(e28, d, start, "morning")
    assert "f_gap_open" not in m.ft_cols and "f_rv_intra" not in m.ft_cols and "d_sess_pm" not in m.ft_cols
    i = m.pre_cols.index("d_sess_pm")
    beta0, a0 = m.start
    assert np.isclose(a0, m.a_pre + m.beta_pre[i] * (0.0 - m.scaler.mean[i]) / m.scaler.std[i])
    pm, _, _ = fit_block(e28, d, start, "afternoon")
    assert np.isclose(pm.start[1], pm.a_pre + pm.beta_pre[i] * (1.0 - pm.scaler.mean[i]) / pm.scaler.std[i])


def test_afternoon_scaler_in_inner_cv_uses_rows_before_each_cut(e28):
    d = frame(n_days=320)
    start = np.sort(d["date"].unique())[300]
    a, _, ft = fit_block(e28, d, start, "afternoon")
    assert a.fold_extra, "ต้องมี fold ของ inner CV"
    for f in a.fold_extra:
        changed = d.copy()
        rows = changed.index.isin(ft.index) & (changed["date"] >= f["cut"]).to_numpy()
        changed.loc[rows, ["f_gap_open", "f_rv_intra"]] = 1000.0
        b, _, _ = fit_block(e28, changed, start, "afternoon")
        same = [g for g in b.fold_extra if g["cut"] == f["cut"]]
        assert same and np.allclose(same[0]["mean"], f["mean"]) and np.allclose(same[0]["std"], f["std"])


def test_full_run_is_locked_before_friday_evening(e28):
    assert not e28.full_run_allowed(pd.Timestamp("2026-11-13 17:59", tz=TZ))
    assert e28.full_run_allowed(pd.Timestamp("2026-11-13 18:00", tz=TZ))
