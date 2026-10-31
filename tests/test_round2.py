"""รอบที่ 2: ตัวชี้วัดขึ้น vs ลง, baseline แบบ updown, ช่วงราคา conformal และ sig20 ที่ใช้อดีตเท่านั้น"""
import numpy as np
import pandas as pd

from forecast.evaluation.metrics import row_losses
from forecast.features.dataset import add_sigma
from forecast.models.baseline import BaselineModel
from forecast.models.e9 import ConformalPrice, E9LR

Q = [0.10, 0.25, 0.50, 0.75, 0.90]


def _pred(direction, p_up):
    n = len(direction)
    d = np.asarray(direction, float)
    return pd.DataFrame({"direction": d, "up": (d == 1).astype(float), "p_up": p_up, "z": 0.0,
                         "p_ref": 100.0, "p_end": 100.0, **{f"q{round(q * 100):02d}": 0.0 for q in Q}},
                        index=range(n))


def test_flat_rows_not_scored_in_nonflat_metrics():
    l = row_losses(_pred([1, 0, -1, 0], [0.7, 0.7, 0.3, 0.3]), Q)
    assert l["err_nf"].tolist()[0] == 0 and np.isnan(l["err_nf"].iat[1]) and l["err_nf"].iat[2] == 0
    assert l["err_fw"].tolist() == [0, 1, 0, 1]            # flat นับผิดเสมอ
    assert np.isnan(l["logloss_ud"].iat[1]) and np.isclose(l["logloss_ud"].iat[0], -np.log(0.7))


def test_updown_baseline_ignores_flat_days():
    meta = pd.DataFrame({"direction": [1, 1, -1, 0, 0, 0]})
    up = (meta["direction"] == 1).astype(float).to_numpy()
    assert np.isclose(BaselineModel(Q).fit(None, up, np.zeros(6)).p_up_, 2 / 6)
    assert np.isclose(BaselineModel(Q, target="updown").fit(None, up, np.zeros(6), meta).p_up_, 2 / 3)


def _meta(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B", tz="Asia/Bangkok")
    sig = np.full(n, 0.01)
    return pd.DataFrame({"date": dates, "z": rng.normal(0, 0.01, n), "sig20": sig,
                         "direction": rng.choice([-1, 0, 1], n), "f_a": rng.normal(size=n)})


def test_conformal_point_is_last_price_and_scales_with_sigma():
    m = _meta()
    cp = ConformalPrice(Q).fit(m)
    zq = cp.predict(pd.DataFrame({"sig20": [0.01, 0.02]}))
    assert np.allclose(zq[:, 2], 0)                        # ราคาจุด = ราคาล่าสุด
    assert np.allclose(zq[1], 2 * zq[0])                   # ช่วงกว้างขึ้นตาม σ̂
    assert zq[0, 0] < 0 < zq[0, 4]


def test_sig20_uses_only_past_z():
    dates = pd.date_range("2025-01-01", periods=40, freq="B", tz="Asia/Bangkok")
    ds = pd.DataFrame({"date": dates, "symbol": "KBANK", "session": "morning",
                       "z": np.random.default_rng(1).normal(0, 0.01, 40)})
    a = add_sigma(ds)
    ds2 = ds.copy()
    ds2.loc[30:, "z"] = 9.0                                # เปลี่ยนอนาคต (แถว 30 เป็นต้นไป)
    b = add_sigma(ds2)
    assert np.allclose(a["sig20"].iloc[:31], b["sig20"].iloc[:31], equal_nan=True)


def test_e9_lr_trains_on_nonflat_only():
    m = _meta(200)
    X = m[["f_a"]]
    model = E9LR(Q, n_bags=3).fit(X, None, m["z"].to_numpy(), m)
    out = model.predict(X, m)
    assert out["p_up"].between(0, 1).all() and np.allclose(out["q50"], 0)
    assert model.C_ in model.Cs


def test_flat_targets():
    from forecast.models.e9 import target_rows
    meta = pd.DataFrame({"direction": [1, 0, -1, np.nan], "p_end": [101.0, 100.0, 99.5, np.nan],
                         "p_ref": [100.0, 100.0, 99.5, 100.0]})
    m, y = target_rows(meta, "flat_dir")
    assert m.tolist() == [True, True, True, False] and y.tolist() == [0, 1, 0]
    m, y = target_rows(meta, "flat_price")
    assert m.tolist() == [True, True, True, False] and y.tolist() == [0, 1, 1]
    m, y = target_rows(meta, "price_updown")                 # เฉพาะแถวที่ปลาย ≠ P(c)
    assert m.tolist() == [True, False, False, False] and y.tolist() == [1]


def test_flat_features_use_only_finished_sessions():
    from forecast.config import load_rules
    from forecast.features.dataset import add_flat_features
    rules = load_rules()
    tz = rules["timezone"]
    days = pd.date_range("2025-06-02", periods=30, freq="B", tz=tz)
    bars = []
    for i, day in enumerate(days):
        for h in (10, 11, 12, 14, 15, 16):
            start = day + pd.Timedelta(hours=h)
            bars.append({"date": day, "bar_start": start, "bar_end": start + pd.Timedelta(hours=1),
                         "open": 100.0, "close": 100.0, "high": 100.0 + i, "low": 100.0})   # ช่วงกว้างขึ้นทุกวัน
    bars = pd.DataFrame(bars)
    rows = []
    for day in days:
        for sess, cut in (("morning", "09:15"), ("afternoon", "12:15")):
            rows.append({"date": day, "symbol": "KBANK", "session": sess,
                         "cutoff_ts": day + pd.Timedelta(cut + ":00"), "p_ref": 100.0, "p_end": 100.0,
                         "r": 0.0, "direction": 0.0, "f_rv_intra": 0.001})
    out = add_flat_features(pd.DataFrame(rows), {"KBANK": bars}, rules).set_index(["date", "session"])
    # วันที่ i (เช้าและบ่าย) เห็นได้แค่รอบบ่ายของวันที่ i−1 (ช่วง = i−1 บาท = 2(i−1) ช่อง ที่ tick 0.5)
    for i in range(1, 30):
        for sess in ("morning", "afternoon"):
            assert out.at[(days[i], sess), "fl_prev_range_ticks"] == 2 * (i - 1)
    assert np.isnan(out.at[(days[0], "morning"), "fl_prev_range_ticks"])
    assert out.loc[(slice(None), "morning"), "fl_rv_intra"].eq(0).all()
