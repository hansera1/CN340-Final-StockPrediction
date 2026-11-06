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


# ---------- ข้อ 4: ช่องโมเดล + เป้าหมาย + รันเงา
def _rules():
    from forecast.config import load_rules
    return load_rules()


def test_defaults_are_a_ud_and_pr_a_for_all_slots():
    from forecast.live.slots import direction_spec, direction_target, live_specs, price_specs
    r = _rules()
    assert r["live"].get("shadow_slots") == []
    for s in ("morning", "afternoon"):
        assert direction_target(r, s) == "up_down"
        assert direction_spec(r, s) == "R4ATC/E12/fd_lr"
        assert price_specs(r, s) == ("R4ATC/E9FLAT/aprice_lr", "R4ATC/E9FLAT/pprice_lr")
    assert set(live_specs(r).values()) == {"R4ATC/E12/fd_lr", "R4ATC/E9FLAT/aprice_lr", "R4ATC/E9FLAT/pprice_lr",
                                           "R4ATC/E9/baseline_ud", "R4ATC/E9FLAT/adir_lr"}


def test_up_notup_uses_own_folder_per_session():
    import copy

    import pytest
    from forecast.live.slots import direction_spec
    r = copy.deepcopy(_rules())
    r["live"]["direction_target"]["afternoon"] = "up_notup"
    assert direction_spec(r, "morning") == "R4ATC/E12/fd_lr"
    assert direction_spec(r, "afternoon") == "R4ATC_UN/E12/fd_lr"
    r["live"]["direction_target"]["morning"] = "x"
    with pytest.raises(ValueError):
        direction_spec(r, "morning")


def test_decide_threshold_by_target():
    from forecast.live.slots import decide
    assert decide(0.5, "up_down") == "ขึ้น" and decide(0.5, "up_notup") == "ลง"
    assert decide(0.51, "up_notup") == "ขึ้น" and decide(0.49, "up_down") == "ลง"


def test_any_sklearn_pipeline_per_slot():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from forecast.live.predict import predict_row
    X = pd.DataFrame({"f_a": [0.0, 1.0, 2.0, 3.0], "f_b": [1.0, 0.0, 1.0, 0.0]})
    pipe = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression()).fit(X, [0, 0, 1, 1])
    bundle = {"kind": "sklearn", "model": pipe, "feature_cols": ["f_a", "f_b"], "group_by": ["session"]}
    p = predict_row(bundle, "KBANK", "morning", None, pd.Timestamp("2026-11-09", tz="Asia/Bangkok"))
    assert 0 < p["p_up"] < 1


def test_shadow_slots_write_nothing_and_do_not_fetch(tmp_path, monkeypatch):
    """รันเงาคืนแค่ log: ไม่แตะ submission.dir / predictions.jsonl และไม่ login Settrade"""
    import copy

    from forecast.config import OUTPUT_DIR, ROOT
    from forecast.data import settrade
    from forecast.live.predict import predict_shadow_slots
    monkeypatch.setattr(settrade, "connect", lambda: (_ for _ in ()).throw(AssertionError("login ระหว่างรันเงา")))
    r = copy.deepcopy(_rules())
    sub_dir = ROOT / r["submission"]["dir"]
    log = OUTPUT_DIR / "logs" / "predictions.jsonl"
    before = (sorted(p.name for p in sub_dir.glob("*")), log.stat().st_mtime if log.exists() else None)
    r["live"]["shadow_slots"] = [{"slot": "afternoon_dir", "spec": "R4ATC/E12/fd_lr", "target": "up_notup"},
                                 {"slot": "afternoon_price", "eq": "R4ATC/E9FLAT/aprice_lr", "up": "R4ATC/E9FLAT/pprice_lr"},
                                 {"slot": "morning_dir", "spec": "R4ATC/E12/fd_lr"}]
    out = predict_shadow_slots("afternoon", "2026-11-06", r, now="2026-11-06 12:35")
    assert len(out) == 4 and {x.slot for x in out} == {"afternoon_dir", "afternoon_price"}
    assert all(x.model_version and x.reference_price > 0 for x in out)
    assert (sorted(p.name for p in sub_dir.glob("*")), log.stat().st_mtime if log.exists() else None) == before

