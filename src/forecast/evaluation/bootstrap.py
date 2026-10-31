"""เทียบสองโมเดลด้วย block bootstrap รายวัน

เก็บทุกหุ้นและทุกรอบของวันเดียวกันไว้ด้วยกัน (ไม่สุ่มรายแถว / ไม่ใช้ binomial เทียบ 50%)
"""
import numpy as np
import pandas as pd

from forecast.evaluation.metrics import row_losses

KEYS = ["date", "symbol", "session"]


def block_bootstrap_diff(pred_a, pred_b, quantiles, metric, block_days=1, n_boot=2000, seed=0):
    """ค่า diff = loss(A) − loss(B) → ติดลบแปลว่า A ดีกว่า"""
    la = pred_a[KEYS].join(row_losses(pred_a, quantiles)[[metric]], how="inner")
    lb = pred_b[KEYS].join(row_losses(pred_b, quantiles)[[metric]], how="inner")
    m = la.merge(lb, on=KEYS, suffixes=("_a", "_b"))
    if m.empty:
        return None
    m["d"] = m[f"{metric}_a"] - m[f"{metric}_b"]
    m = m.dropna(subset=["d"])            # ตัวชี้วัดที่ไม่นับบางแถว (เช่น err_nf ในวัน flat)
    if m.empty:
        return None
    daily = m.groupby("date")["d"].agg(["sum", "count"]).sort_index()
    s, c = daily["sum"].to_numpy(), daily["count"].to_numpy()
    n = len(daily)
    b = min(block_days, n)
    n_blocks = int(np.ceil(n / b))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - b + 1, n_blocks)
        idx = (starts[:, None] + np.arange(b)).ravel()[:n]
        diffs[i] = s[idx].sum() / c[idx].sum()

    return {
        "metric": metric,
        "n_days": n,
        "n_rows": len(m),
        "mean_diff": s.sum() / c.sum(),
        "ci_low": np.quantile(diffs, 0.025),
        "ci_high": np.quantile(diffs, 0.975),
        "prob_a_better": float((diffs < 0).mean()),
    }


def compare(pred_a, pred_b, quantiles, metrics=("logloss", "brier", "abs_err_baht", "pinball"), **kw):
    rows = [block_bootstrap_diff(pred_a, pred_b, quantiles, m, **kw) for m in metrics]
    return pd.DataFrame([r for r in rows if r])
