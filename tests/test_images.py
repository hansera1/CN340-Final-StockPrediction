"""E7: ภาพต้องสร้างจากแท่งที่จบก่อนจุดตัดเท่านั้น และหัวทำนายรันได้"""
import numpy as np
import pandas as pd
import pytest

from forecast.config import session_times
from forecast.data.io import load_bars
from forecast.features.images import build_images, morlet_cwt
from forecast.models.image_heads import FrozenHeadModel, FrozenLRModel


@pytest.mark.parametrize("kind", ["scalogram", "candlestick"])
def test_image_ignores_bars_after_cutoff(rules, raw_dir, kind):
    bars = load_bars("KBANK", rules, raw_dir)
    date = bars["date"].iloc[-1]
    rows = pd.DataFrame({"date": [date, date], "session": ["morning", "afternoon"]})
    base = build_images(bars, rows, rules, kind)
    moved = bars.copy()
    cut = session_times(rules, "afternoon", date)["cutoff"]
    after = moved["bar_end"] > cut
    moved.loc[after, ["open", "high", "low", "close"]] *= 1.5
    moved.loc[after, "volume"] *= 9
    np.testing.assert_array_equal(base, build_images(moved, rows, rules, kind))
    assert base.shape == (2, 3, 224, 224) and np.isfinite(base).all()
    assert not np.allclose(base[0], base[1])      # รอบบ่ายเห็นแท่งเช้าวันนี้เพิ่ม → ภาพต่างจากรอบเช้า


def test_morlet_finds_frequency():
    t = np.arange(56)
    slow, fast = morlet_cwt(np.sin(2 * np.pi * t / 20)), morlet_cwt(np.sin(2 * np.pi * t / 4))
    assert slow.mean(axis=1).argmax() > fast.mean(axis=1).argmax()   # คาบยาว → scale ใหญ่กว่า


@pytest.mark.parametrize("cls", [FrozenLRModel, FrozenHeadModel])
def test_heads_fit_predict(cls):
    rng = np.random.default_rng(0)
    n = 300
    emb = rng.normal(size=(n, 512)).astype(np.float32)
    up = (emb[:, 0] + rng.normal(0, 1, n) > 0).astype(float)
    meta = pd.DataFrame({"date": np.repeat(pd.bdate_range("2025-01-01", periods=n // 2), 2)})
    X = pd.DataFrame({"a": np.zeros(n)})
    kw = {"seeds": (0, 1), "epochs": 20} if cls is FrozenHeadModel else {}
    m = cls([0.1, 0.25, 0.5, 0.75, 0.9], **kw).fit(X, up, rng.normal(0, 0.01, n), meta, emb)
    p = m.predict(X, meta, emb)
    assert p["p_up"].between(0, 1).all()
    assert np.corrcoef(p["p_up"], emb[:, 0])[0, 1] > 0.3          # จับสัญญาณใน feature แรกได้
    if cls is FrozenHeadModel:
        assert {"p_up_seed0", "p_up_seed1"} <= set(p.columns)
