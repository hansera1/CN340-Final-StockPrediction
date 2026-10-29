"""ให้คะแนนคำทำนายจริงหลังรู้ผล (เย็นทุกวัน และสรุปสัปดาห์ 14 พ.ย.)

    python scripts/06_score_live.py [--raw-dir ...]
    → outputs/reports/live_scored.csv, live_summary.csv
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules
from forecast.data.io import load_bars
from forecast.evaluation.metrics import summarize
from forecast.labels.make_labels import make_labels
from forecast.live.log_schema import read_logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path)
    args = ap.parse_args()
    rules = load_rules()

    logs = pd.DataFrame(read_logs(OUTPUT_DIR / "logs" / "predictions.jsonl"))
    # ถ้าส่งซ้ำรอบเดียวกัน ใช้รายการแรกที่ส่งก่อนกำหนด (ไม่เลือกอันที่ผลดีกว่า)
    logs = logs.sort_values("generated_at").drop_duplicates(["date", "symbol", "session"], keep="first")
    logs["date"] = pd.to_datetime(logs["date"]).dt.tz_localize(rules["timezone"])
    zq = pd.json_normalize(logs["z_quantiles"]).set_index(logs.index)
    preds = logs[["date", "symbol", "session", "p_up", "model_version", "fallback_reason"]].join(zq)
    preds["p_ref"] = logs["reference_price"]

    labels = pd.concat([make_labels(load_bars(s, rules, args.raw_dir), rules) for s in rules["symbols"]])
    scored = preds.merge(labels[["date", "symbol", "session", "p_end", "r", "direction", "up"]],
                         on=["date", "symbol", "session"], how="left")
    scored["z"] = np.log(scored["p_end"] / scored["p_ref"])

    out = OUTPUT_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "live_scored.csv", index=False, encoding="utf-8-sig")
    done = scored[scored["up"].notna()]
    print(f"คำทำนาย {len(scored)} รายการ, รู้ผลแล้ว {len(done)}")
    if len(done):
        s = summarize(done, rules["quantiles"])
        s.to_csv(out / "live_summary.csv", index=False)
        print(s[["group", "n", "n_flat", "acc_flat_as_wrong", "acc_flat_excluded", "logloss", "mae_baht", "mae_naive_baht",
                 "coverage50", "coverage80"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
