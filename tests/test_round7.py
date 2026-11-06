"""รอบ 7 ข้อ 2: ตัวชี้วัดและเป้าหมายขึ้น/ไม่ขึ้น"""
import numpy as np
import pandas as pd

from forecast.evaluation.metrics import row_losses, summarize
from forecast.models.baseline import BaselineModel
from forecast.models.e9 import target_rows

Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def frame():
    d = pd.DataFrame({
        "symbol": "KBANK", "session": "morning",
        "direction": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, np.nan],
        "p_up": [0.7, 0.4, 0.4, 0.7, 0.4, 0.7, 0.9],
        "up": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "z": 0.0, "p_ref": 100.0, "p_end": [101.0, 101.0, 99.0, 99.0, 100.0, 100.0, 100.0]})
    for q in Q:
        d[f"q{round(q * 100):02d}"] = 0.0
    return d


def test_acc_up_notup_counts_flat_as_not_up():
    l = row_losses(frame(), Q)
    # ขึ้น+ทายขึ้น ถูก · ขึ้น+ทายลง ผิด · ลง+ทายลง ถูก · ลง+ทายขึ้น ผิด · นิ่ง+ทายลง ถูก · นิ่ง+ทายขึ้น ผิด · ไม่มี label ไม่นับ
    assert l["correct_un"].iloc[:6].tolist() == [1, 0, 1, 0, 1, 0]
    assert np.isnan(l["correct_un"].iloc[6])


def test_threshold_is_strictly_above_half():
    d = frame().iloc[[0]].copy()
    d["p_up"] = 0.5
    assert row_losses(d, Q)["correct_un"].iat[0] == 0


def test_logloss_and_brier_use_up_notup_label():
    l = row_losses(frame(), Q)
    assert np.isclose(l["logloss_un"].iat[4], -np.log(0.6))      # นิ่ง: y = 0, p = 0.4
    assert np.isclose(l["logloss_un"].iat[0], -np.log(0.7))      # ขึ้น: y = 1
    assert np.isclose(l["brier_un"].iat[5], 0.49)                # นิ่ง: y = 0, p = 0.7


def test_summarize_reports_new_metrics():
    s = summarize(frame().iloc[:6].assign(date=pd.Timestamp("2026-01-05")), Q)
    r = s[s["group"] == "ALL"].iloc[0]
    assert np.isclose(r["acc_up_notup"], 0.5)
    assert np.isclose(r["rmse_baht"], np.sqrt(4 / 6))


def test_target_up_notup_trains_on_every_labelled_day():
    d = frame()
    m, y = target_rows(d, "up_notup")
    assert m.tolist() == [True] * 6 + [False]
    assert y.tolist() == [1, 1, 0, 0, 0, 0]


def test_baseline_up_notup_is_share_of_up_days():
    d = frame().iloc[:6]
    b = BaselineModel(Q, target="up_notup").fit(pd.DataFrame(index=d.index), d["up"], d["z"], d)
    p = b.predict(pd.DataFrame(index=d.index))["p_up"]
    assert np.allclose(p, 2 / 6)
