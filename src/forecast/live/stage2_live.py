"""E13 ชั้น 2 ในวันจริง (results/e13_preregistration.md) — ตารางฝึกใช้ร่วมกับ backtest (experiments/21_e13_stage2.py)

ตารางฝึก: [date, symbol, p1, direction, f5_*] ของรอบบ่าย
  p₁ out-of-sample: E12/fd_lr walk-forward + E12_holdout + log ส่งจริง (เฉพาะแถวที่ trained_until < date; 30 ต.ค. ไม่มี)
  f5: yfinance 5m (แหล่งเดียวกับ backtest) · label: ทิศทางรอบบ่ายจากไฟล์หลัก
ฝึกตอนเย็นวันก่อน → outputs/models/stage2/<วันที่ทาย>.json · รอบบ่าย: Settrade 1m วันนี้ → 5m → f5 → p_final
"""
import json

import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, raw_source_dir
from forecast.data.io import aggregate_bars, load_bars
from forecast.features.fine import F5, f5_features, f5_table
from forecast.labels.make_labels import labels_for
from forecast.live.log_schema import read_logs
from forecast.models.stage2 import Stage2, fit_for_day

EXP = OUTPUT_DIR / "experiments"
STAGE2_DIR = OUTPUT_DIR / "models" / "stage2"


def _load(exp, variant, tz):
    p = pd.read_csv(EXP / exp / f"{variant}_predictions.csv")
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def p1_backtest(rules):
    """p₁ out-of-sample รอบบ่าย: walk-forward (ถึง 12 ต.ค.) + holdout (14–29 ต.ค.) พร้อม p_e0 และ label"""
    tz = rules["timezone"]
    parts = []
    for exp in ["E12", "E12_holdout"]:
        p = _load(exp, "fd_lr", tz)
        e0 = _load(exp, "baseline_ud", tz)[["date", "symbol", "session", "p_up"]].rename(columns={"p_up": "p_e0"})
        p = p.merge(e0, on=["date", "symbol", "session"], how="left").assign(p1_source=exp)
        parts.append(p[p["session"] == "afternoon"])
    t = pd.concat(parts, ignore_index=True).rename(columns={"p_up": "p1"})
    return t[["date", "symbol", "p1", "p_e0", "direction", "p1_source"]].sort_values(["date", "symbol"])


def p1_live(rules, log=None):
    """p₁ จาก log ส่งจริงรอบบ่าย (รายการแรกต่อรอบ) เฉพาะโมเดลที่ trained_until < date (กันแถว in-sample)"""
    log = log or OUTPUT_DIR / "logs" / "predictions.jsonl"
    if not log.exists():
        return pd.DataFrame(columns=["date", "symbol", "p1", "p1_source"])
    d = pd.DataFrame(read_logs(log))
    d = d[(d["session"] == "afternoon") & d["feature_version"].str.startswith("trained_until=")]
    d = d.sort_values("generated_at").drop_duplicates(["date", "symbol"], keep="first")
    trained = pd.to_datetime(d["feature_version"].str.replace("trained_until=", "", regex=False))
    date = pd.to_datetime(d["date"])
    d = d[trained < date]
    p1 = d["p_stage1"].fillna(d["p_up"]) if "p_stage1" in d else d["p_up"]
    return pd.DataFrame({"date": pd.to_datetime(d["date"]).dt.tz_localize(rules["timezone"]), "symbol": d["symbol"],
                         "p1": p1.astype(float), "p1_source": "live log"})


def f5_history(rules, sym, placebo=False):
    """f5 จาก yfinance 5m (+ placebo = f5 ของวันที่มีแท่ง 5m ก่อนหน้า)"""
    f = f5_table(load_bars(sym, rules, raw_source_dir(rules, "yfinance"), "5m"), rules).sort_values("date")
    if placebo:
        f = pd.concat([f, f[F5].shift(1).add_prefix("plc_")], axis=1)
    return f.assign(symbol=sym)


def afternoon_labels(rules, sym):
    lab = labels_for(load_bars(sym, rules), rules, rules["data"]["interval"])
    return lab.loc[lab["session"] == "afternoon", ["date", "symbol", "direction"]]


def training_table(rules):
    """ตารางฝึกสำหรับวันจริง: p₁ backtest + log ส่งจริง, label ปัจจุบัน, f5 จาก yfinance 5m"""
    bt = p1_backtest(rules)[["date", "symbol", "p1", "p1_source"]]
    live = p1_live(rules)
    p1 = pd.concat([bt, live[~live.set_index(["date", "symbol"]).index.isin(bt.set_index(["date", "symbol"]).index)]])
    lab = pd.concat([afternoon_labels(rules, s) for s in rules["symbols"]])
    f5 = pd.concat([f5_history(rules, s) for s in rules["symbols"]])
    return p1.merge(lab, on=["date", "symbol"], how="left").merge(f5, on=["date", "symbol"], how="left")


def train_for(rules, date):
    """ฝึกชั้น 2 สำหรับวันที่ทาย date (ข้อมูลก่อน date เท่านั้น) → บันทึก json → info"""
    cfg = rules["live"]["stage2"]
    date = pd.Timestamp(date).tz_localize(rules["timezone"]) if pd.Timestamp(date).tzinfo is None else pd.Timestamp(date)
    m, info = fit_for_day(training_table(rules), date, cfg["train_days"], cfg["l2"], cfg["min_nonflat"])
    out = {**info, "train_days": cfg["train_days"], "model": m.to_dict() if m else None, "features": F5}
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGE2_DIR / f"{date:%Y-%m-%d}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out, path


def apply(rules, date, p1, today_1m):
    """รอบบ่าย → (p_final, betas, เหตุผล) · ทุกปัญหา → p_final = p₁ พร้อมเหตุผล"""
    path = STAGE2_DIR / f"{pd.Timestamp(date):%Y-%m-%d}.json"
    if not path.exists():
        return p1, None, f"ไม่มีไฟล์ β ชั้น 2 ({path.name})"
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["model"] is None:
        return p1, None, d["reason"]
    if today_1m is None or not len(today_1m):
        return p1, None, "ไม่มีแท่ง 1m วันนี้ (หรือช้าเกิน max_bar_lag_min)"
    f = f5_features(aggregate_bars(today_1m, 5, rules), date, rules)
    x = np.array([[f[k] for k in F5]], float)
    if not np.isfinite(x).all():
        return p1, None, "f5 ไม่ครบ"
    m = Stage2.from_dict(d["model"])
    return float(m.predict(x, [p1])[0]), dict(zip(F5, m.beta_.tolist())), None
