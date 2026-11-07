"""E23: skip เริ่มจาก P(c) และการฝึกใช้เฉพาะแถวฝึก"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT

torch = pytest.importorskip("torch")
DEV = torch.device("cpu")


@pytest.fixture(scope="module")
def e23():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("48_e23_skip_fx")


def data(n_days=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2026-01-05", periods=n_days).to_numpy(), 2)
    n = len(dates)
    return (rng.random((n, 3, 16, 16)).astype(np.float32), np.tile([0.0, 1.0], n_days), rng.normal(size=(n, 4)), rng.normal(0, 0.01, n), dates)


@pytest.mark.parametrize("use_cnn", [False, True])
def test_untrained_net_answers_the_reference_price(e23, use_cnn):
    net = e23.make_net(use_cnn, 4)
    out = net(torch.zeros(3, 3, 16, 16) if use_cnn else None, torch.zeros(3, 1), torch.randn(3, 4))
    assert torch.equal(out, torch.zeros(3))


@pytest.mark.parametrize("use_cnn", [False, True])
def test_later_rows_do_not_change_predictions(e23, use_cnn):
    img, sym, tab, z, dates = data()
    days = np.unique(dates)
    tr, te = dates < days[20], (dates >= days[20]) & (dates < days[22])
    a = e23.fit_predict(use_cnn, img, sym, tab, z, tr, te, dates, DEV, (0,), 2, "f", [])
    img2, tab2, z2 = img.copy(), tab.copy(), z.copy()
    later = dates >= days[22]
    img2[later], tab2[later], z2[later] = 9.0, 99.0, 5.0
    b = e23.fit_predict(use_cnn, img2, sym, tab2, z2, tr, te, dates, DEV, (0,), 2, "f", [])
    assert np.allclose(a, b)


def test_test_day_answer_does_not_enter_training(e23):
    img, sym, tab, z, dates = data()
    days = np.unique(dates)
    tr, te = dates < days[20], dates == days[20]
    a = e23.fit_predict(False, img, sym, tab, z, tr, te, dates, DEV, (0,), 2, "f", [])
    z2 = z.copy()
    z2[te] = 5.0
    assert np.allclose(a, e23.fit_predict(False, img, sym, tab, z2, tr, te, dates, DEV, (0,), 2, "f", []))


def test_points_apply_the_shift_to_the_reference_price(e23):
    k = {"date": ["a"], "symbol": ["KBANK"], "session": ["morning"]}
    x = pd.DataFrame({**k, "p_ref": [100.0], "p_end": [101.0]})
    dz = pd.DataFrame({**k, "dz_skip_tab": [0.01], "dz_skip_tab_fx": [0.0], "dz_skip_cnn": [-0.01], "dz_skip_cnn_fx": [np.nan]})
    p = e23.add_points(x, dz).iloc[0]
    assert np.isclose(p["pt_skip_tab"], 100 * np.exp(0.01)) and p["pt_skip_tab_fx"] == 100.0 and np.isnan(p["pt_skip_cnn_fx"])


def test_verdict_uses_the_preregistered_representative(e23):
    rows = []
    for m, th in [("skip_cnn_fx", "เช้า"), ("skip_cnn", "บ่าย")]:
        for g in (th, f"KBANK {th}", f"ADVANC {th}"):
            rows.append({"model": m, "group": g, "d_rmse_pra_hi": -0.1, "d_rmse_pra_lo": -0.2, "d_mae_pra_lo": -0.1})
    t = pd.DataFrame(rows)
    t.loc[(t["model"] == "skip_cnn") & (t["group"] == "บ่าย"), "d_rmse_pra_hi"] = 0.1
    v = e23.verdict(t).set_index("session")["pass"]
    assert v["เช้า"] and not v["บ่าย"]
