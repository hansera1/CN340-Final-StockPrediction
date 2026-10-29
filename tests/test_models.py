import numpy as np
import pytest

from forecast.evaluation.bootstrap import compare
from forecast.evaluation.metrics import round_to_tick, summarize, tick_size
from forecast.evaluation.runner import run_variant
from forecast.features.dataset import assemble_dataset


@pytest.fixture
def ds(rules, raw_dir):
    return assemble_dataset(rules, raw_dir=raw_dir, ext_dir=raw_dir / "none")


@pytest.mark.parametrize("model", ["baseline", "independent", "cascade"])
def test_walk_forward_runs_and_holds_out_tail(rules, ds, model):
    p = run_variant(ds, {"name": model, "model": model}, rules)
    holdout = np.sort(ds["date"].unique())[-rules["evaluation"]["holdout_days"]:]
    assert not p["date"].isin(holdout).any()
    qs = p[["q10", "q25", "q50", "q75", "q90"]].to_numpy()
    assert (np.diff(qs, axis=1) >= 0).all()                 # ไม่มี quantile crossing
    s = summarize(p, rules["quantiles"])
    assert s.loc[s["group"] == "ALL", "n"].iat[0] == p["up"].notna().sum()


def test_pooled_and_resampled_variants(rules, raw_dir):
    ds5 = assemble_dataset(rules, raw_dir=raw_dir, ext_dir=raw_dir / "none", feature_minutes=5)
    p = run_variant(ds5, {"name": "pooled", "model": "independent", "group_by": ["session"]}, rules)
    assert set(p["symbol"]) == set(rules["symbols"])


def test_bootstrap_compare(rules, ds):
    a = run_variant(ds, {"name": "b", "model": "baseline"}, rules)
    b = run_variant(ds, {"name": "i", "model": "independent"}, rules)
    c = compare(b, a, rules["quantiles"], n_boot=50)
    assert set(c["metric"]) == {"logloss", "brier", "abs_err_baht", "pinball"}
    assert (c["ci_low"] <= c["ci_high"]).all()


def test_set_tick_sizes():
    assert tick_size(150) == 0.5 and tick_size(300) == 1.0 and tick_size(24.9) == 0.1
    assert round_to_tick(151.3) == 151.5
