"""T5 (NEXT_STEPS รอบที่ 4 ข้อ 5) — สำหรับรายงาน ไม่ใช้เปลี่ยนโมเดล

    python experiments/19_t5_xd.py
→ results/round4/t5_xd.csv

วัน XD 2024–2026 จาก data/external/calendar.csv (yfinance dividends = ex-date)
1. MAE รอบเช้าในวัน XD เทียบวันปกติ: ราคาจุด = P(c) แบบใหม่ (ราคาปิดทางการเมื่อวาน ไม่หักปันผล — แบบ backtest)
   และ P(c) − ปันผล (กฎวันจริงข้อ 5)
2. ความแม่น fd_lr (ไม่นับ flat) ใน 10 วันทำการแรกนับจากวัน XD เทียบวันอื่น
แถวทดสอบ walk-forward เดิม (ไม่แตะ holdout)
"""
import importlib.util

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import EXTERNAL_DIR, ROOT, load_rules
from forecast.evaluation.metrics import round_to_tick

_spec = importlib.util.spec_from_file_location("s18", ROOT / "experiments" / "18_score_round4.py")
s18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s18)


def main():
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    cal = pd.read_csv(EXTERNAL_DIR / "calendar.csv", parse_dates=["date"])
    xd = cal[cal["event"] == "XD"]
    rows = []

    p = s18.load("R4PCOFF", "E9_baseline_ud", tz)
    p = p[(p["session"] == "morning") & p["p_end"].notna()].copy()
    p["day"] = p["date"].dt.tz_localize(None).dt.normalize()
    p = p.merge(xd[["date", "symbol", "amount"]].rename(columns={"date": "day"}), on=["day", "symbol"], how="left")
    is_xd = p["amount"].notna()
    adj = np.where(is_xd, round_to_tick(p["p_ref"] - p["amount"].fillna(0)), p["p_ref"])
    e_pc, e_adj = (p["p_end"] - p["p_ref"]).abs(), (p["p_end"] - adj).abs()
    for sym in [None, *rules["symbols"]]:
        m = p["symbol"].eq(sym) if sym else pd.Series(True, index=p.index)
        rows.append({"test": "MAE เช้า วัน XD vs ปกติ", "symbol": sym or "ALL", "n_xd": int((m & is_xd).sum()),
                     "n_other": int((m & ~is_xd).sum()), "xd_mae_pc": e_pc[m & is_xd].mean(),
                     "xd_mae_pc_minus_div": e_adj[m & is_xd].mean(), "other_mae_pc": e_pc[m & ~is_xd].mean(),
                     "xd_days": ", ".join(p.loc[m & is_xd, "day"].dt.strftime("%Y-%m-%d"))})

    d = s18.load("E12", "fd_lr", tz)
    d = d[d["direction"].notna() & (d["direction"] != 0)].copy()
    d["day"] = d["date"].dt.tz_localize(None).dt.normalize()
    days = sorted(d["day"].unique())
    after = set()
    for _, x in xd.iterrows():
        i = np.searchsorted(days, x["date"])
        after |= {(days[j], x["symbol"]) for j in range(i, min(i + 10, len(days)))}
    in_win = pd.Series([(a, b) in after for a, b in zip(d["day"], d["symbol"])], index=d.index)
    hit = ((d["p_up"] >= 0.5) == (d["direction"] == 1)).astype(float)
    for sym in [None, *rules["symbols"]]:
        m = d["symbol"].eq(sym) if sym else pd.Series(True, index=d.index)
        x = pd.Series(np.where(m & in_win, hit, np.nan))
        y = pd.Series(np.where(m & ~in_win, hit, np.nan))
        _, lo, hi = s18.boot_mean(x, d["date"], n_boot)
        rows.append({"test": "fd_lr 10 วันหลัง XD vs อื่น", "symbol": sym or "ALL", "n_xd": int((m & in_win).sum()),
                     "n_other": int((m & ~in_win).sum()), "acc_after_xd": x.mean(), "acc_after_lo": lo,
                     "acc_after_hi": hi, "acc_other": y.mean()})
    t = pd.DataFrame(rows)
    s18.OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(s18.OUT / "t5_xd.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.drop(columns="xd_days").round(4).to_string(index=False))
    print(t.loc[t["xd_days"].notna(), ["symbol", "xd_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
