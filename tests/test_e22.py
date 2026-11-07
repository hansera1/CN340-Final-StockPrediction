"""E22: k = 0 คือราคาอ้างอิง, k = 1 คือ PR-mean และการเลือก k ใช้เฉพาะวันก่อนบล็อก"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.config import ROOT


@pytest.fixture(scope="module")
def e22():
    sys.path.insert(0, str(ROOT / "experiments"))
    return importlib.import_module("47_e22_shrink_k")


def rows(n_days=120, ez=0.01, follow=1.0, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-06", periods=n_days).to_numpy()
    p = np.full(n_days, 100.0)
    e = np.full(n_days, ez)
    return pd.DataFrame({"date": days, "p_ref": p, "ez": e, "p_end": p * np.exp(follow * e) + rng.normal(0, 0.01, n_days)})


def test_k_zero_is_reference_and_k_one_is_pr_mean(e22):
    assert e22.shrink_point(100.0, 0.02, 0.0) == 100.0
    assert np.isclose(e22.shrink_point(100.0, 0.02, 1.0), 100.0 * np.exp(0.02))


def test_expected_z_formula(e22):
    assert np.isclose(e22.expected_z(0.2, 0.5, 0.02, -0.01), 0.8 * 0.5 * 0.02 + 0.8 * 0.5 * -0.01)


@pytest.mark.parametrize("follow,expected", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_select_k_finds_the_k_that_minimises_rmse(e22, follow, expected):
    r = rows(follow=follow)
    assert e22.select_k(r["p_ref"].to_numpy(), r["p_end"].to_numpy(), r["ez"].to_numpy(), r["date"].to_numpy()) == expected


def test_too_few_days_uses_reference(e22):
    r = rows(n_days=e22.MIN_OOS_DAYS - 1)
    assert e22.select_k(r["p_ref"].to_numpy(), r["p_end"].to_numpy(), r["ez"].to_numpy(), r["date"].to_numpy()) == 0.0


def test_tie_picks_smaller_k(e22):
    r = rows()
    r["ez"] = 0.0
    assert e22.select_k(r["p_ref"].to_numpy(), r["p_end"].to_numpy(), r["ez"].to_numpy(), r["date"].to_numpy()) == 0.0


def test_future_rows_do_not_change_earlier_k(e22):
    r = rows(n_days=150, follow=1.0)
    days = r["date"].to_numpy()
    blocks = [days[i:i + 5] for i in range(60, 150, 5)]
    a = e22.walk_forward_k(r, blocks)
    changed = r.copy()
    cut = days[100]
    changed.loc[changed["date"] >= cut, "p_end"] = 100.0
    b = e22.walk_forward_k(changed, blocks)
    early = r["date"] <= cut
    assert a[early].equals(b[early])
    assert (a[r["date"] < days[60]].isna()).all()


def test_verdict_needs_every_criterion(e22):
    base = []
    for name in ["PR-k"]:
        for _, th in e22.SESSIONS:
            for g in (th, f"KBANK {th}", f"ADVANC {th}"):
                base.append({"model": name, "group": g, "d_rmse_pra_hi": -0.1, "d_rmse_pra_lo": -0.2, "d_mae_pra_lo": -0.1})
    t = pd.DataFrame(base)
    t.loc[t["group"] == "บ่าย", "d_rmse_pra_hi"] = 0.01
    v = e22.verdict(t).set_index("session")["pass"]
    assert v["เช้า"] and not v["บ่าย"]
