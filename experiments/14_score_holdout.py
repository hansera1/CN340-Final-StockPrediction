"""ข้อ 4: คิดคะแนน holdout (10 วันสุดท้าย) ของโมเดลที่ล็อก — รันครั้งเดียวหลังเขียน results/lock_decision.md

ขั้นตอน (ครั้งเดียว ห้ามปรับอะไรหลังเห็นผล):
    python experiments/03_run_experiment.py E12 --holdout --only baseline_ud,fd_lr
    python experiments/03_run_experiment.py E9 --holdout --only lrx            (ตัวเทียบเท่านั้น ไม่ใช้เลือก)
    python experiments/03_run_experiment.py E9FLAT --holdout --only aprice_lr,pprice_lr
    python experiments/14_score_holdout.py
→ results/holdout/*.csv

ทิศทาง: E12 fd_lr (ล็อก) เทียบ E0-ud และ E9-LRx (ความแม่นไม่นับ flat, flat = ผิด, log loss ขึ้น/ลง + block bootstrap รายวัน)
ราคา: กฎค่ามัธยฐาน (A-price LR + p_price) เทียบราคาจุด = P(c), ช่วงราคา = quantile ของ z (E0) รอบราคาจุด
"""
import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_rules
from forecast.evaluation.metrics import tick_size

import importlib.util
_spec = importlib.util.spec_from_file_location("s09", ROOT / "experiments" / "09_score_round2.py")
s09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s09)

OUT = ROOT / "results" / "holdout"
KEYS = ["date", "symbol", "session"]


def load(exp, name, tz):
    p = pd.read_csv(OUTPUT_DIR / "experiments" / f"{exp}_holdout" / f"{name}_predictions.csv")
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def main():
    rules = load_rules()
    tz, q, ev = rules["timezone"], rules["quantiles"], rules["evaluation"]
    OUT.mkdir(parents=True, exist_ok=True)
    e0, lrx, fd = load("E12", "baseline_ud", tz), load("E9", "lrx", tz), load("E12", "fd_lr", tz)
    days = sorted(fd["date"].dt.date.unique())
    print(f"holdout {days[0]} – {days[-1]} ({len(days)} วัน, {len(fd)} รอบ)")

    rows = []
    for name, p in [("E0-ud", e0), ("E9-LRx (ตัวเทียบ)", lrx), ("E12 fd_lr (ล็อก)", fd)]:
        for g in s09.GROUPS:
            row = {"model": name, "group": s09.gname(g), **s09.scores(s09.subset(p, g), q)}
            if name != "E0-ud":
                row.update({f"vsE0ud_{k}": v for k, v in s09.boot(p, e0, q, ev, g).items()})
            if name.startswith("E12"):
                row.update({f"vsLRx_{k}": v for k, v in s09.boot(p, lrx, q, ev, g).items()})
            rows.append(row)
    direction = pd.DataFrame(rows)
    direction.to_csv(OUT / "holdout_direction.csv", index=False, encoding="utf-8-sig")

    # ราคา: กฎค่ามัธยฐาน
    eq, pp = load("E9FLAT", "aprice_lr", tz), load("E9FLAT", "pprice_lr", tz)
    m = fd[KEYS + ["p_ref", "p_end"]].merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "p_eq"}), on=KEYS) \
        .merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "p_price"}), on=KEYS) \
        .merge(e0[KEYS + [c for c in e0 if c.startswith("q") and c[1:].isdigit()]], on=KEYS)
    tick = tick_size(m["p_ref"]).astype(float)
    up, dn = (1 - m["p_eq"]) * m["p_price"], (1 - m["p_eq"]) * (1 - m["p_price"])
    m["shift"] = np.where(up > 0.5, 1, np.where(dn > 0.5, -1, 0))
    m["point"] = m["p_ref"] + m["shift"] * tick
    m["err_rule"] = (m["p_end"] - m["point"]).abs()
    m["err_pc"] = (m["p_end"] - m["p_ref"]).abs()
    for lo, hi, lab in [("q25", "q75", "50"), ("q10", "q90", "80")]:
        m[f"cov{lab}"] = ((m["p_end"] >= m["point"] * np.exp(m[lo] - m["q50"])) &
                          (m["p_end"] <= m["point"] * np.exp(m[hi] - m["q50"]))).astype(float)
    m.to_csv(OUT / "holdout_price_rows.csv", index=False, encoding="utf-8-sig")
    prow = []
    for g in s09.GROUPS:
        s = s09.subset(m, g)
        prow.append({"group": s09.gname(g), "n": len(s), "share_shifted": (s["shift"] != 0).mean(),
                     "mae_rule_baht": s["err_rule"].mean(), "mae_pc_baht": s["err_pc"].mean(),
                     "mae_rule_ticks": (s["err_rule"] / tick[s.index]).mean(),
                     "coverage50": s["cov50"].mean(), "coverage80": s["cov80"].mean()})
    price = pd.DataFrame(prow)
    price.to_csv(OUT / "holdout_price.csv", index=False, encoding="utf-8-sig")
    print(direction[["model", "group", "n", "n_nonflat", "acc_nf", "acc_fw", "logloss_ud"] +
                    [c for c in direction if c.startswith(("vsE0ud_d_acc", "vsLRx_d_acc_nf"))]].round(4).to_string(index=False))
    print(price.round(4).to_string(index=False))
    print("→", OUT)


if __name__ == "__main__":
    main()
