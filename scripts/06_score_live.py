"""ให้คะแนนคำทำนายจริงหลังรู้ผล (เย็นทุกวัน และสรุปสัปดาห์ 14 พ.ย.)

    python scripts/06_score_live.py [--raw-dir ...]
    → outputs/reports/live_scored.csv, live_summary.csv, live_direction_rules.csv

รอบ 7: live_direction_rules.csv แยกตามเป้าหมายที่ส่ง (direction_target) ทุกแถวให้คะแนนครบ 3 กติกา + log loss / Brier ทั้งสองป้าย
  + MAE / RMSE ของราคาที่ส่งจริง (point_price) เทียบ P(c) · acc_up_notup = วันนิ่งนับเป็นไม่ขึ้น (ทาย ลง แล้วนิ่ง = ถูก, ทาย ขึ้น แล้วนิ่ง = ผิด)

ข้อ 1ก รอบที่ 4: คะแนนทิศทางที่ส่งจริง (คอลัมน์ direction ใน log รวม "เท่าเดิม" ถ้าเปิดกฎ 3 คำตอบ) ตามกติกาวัน flat ทุกแบบ
  flat_as_wrong = วัน flat ผิดเสมอ · flat_as_correct = วัน flat ถูกเสมอ · flat_excluded = ไม่นับวัน flat
  exact_3state = ต้องตรง 3 สถานะ (ใช้เมื่อส่ง "เท่าเดิม" ได้)
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules
from forecast.data.io import load_bars
from forecast.evaluation.metrics import summarize
from forecast.config import raw_source_dir
from forecast.labels.make_labels import labels_for
from forecast.live.log_schema import read_logs


def score_labels(sym, rules, raw_dir=None):
    """คำตอบจริง: Settrade 1m (นิยามตรงที่สุด) ก่อน ส่วนวันที่ไม่มีใช้ไฟล์หลัก + open รายวัน"""
    main = labels_for(load_bars(sym, rules, raw_dir), rules, rules["data"]["interval"])
    if raw_dir:
        return main
    try:
        m1 = labels_for(load_bars(sym, rules, raw_source_dir(rules, "settrade"), "1m"), rules, "1m")
    except FileNotFoundError:
        return main
    keys = ["date", "symbol", "session"]
    return pd.concat([m1, main.merge(m1[keys], on=keys, how="left", indicator=True)
                      .query("_merge == 'left_only'").drop(columns="_merge")], ignore_index=True)


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
    preds["answer"] = logs["direction"].map({"ขึ้น": 1, "ลง": -1, "ไม่ขึ้น": -1, "เท่าเดิม": 0})   # "ลง" = log ก่อน 8 พ.ย.
    preds["p_ref"] = logs["reference_price"]
    preds["point_price"] = logs["point_price"]
    preds["direction_target"] = logs["direction_target"] if "direction_target" in logs else "up_down"
    preds["direction_target"] = preds["direction_target"].fillna("up_down")

    labels = pd.concat([score_labels(s, rules, args.raw_dir) for s in rules["symbols"]])
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
        rows = []
        groups = [((t, "ALL", "ALL"), g) for t, g in done.groupby("direction_target")]
        groups += list(done.groupby(["direction_target", "symbol", "session"]))
        for key, g in groups:
            hit, flat = (g["answer"] == g["direction"]), g["direction"] == 0
            p, yud, yun = g["p_up"].clip(1e-4, 1 - 1e-4), g["direction"] == 1, g["direction"] == 1
            ll = lambda y, m: -(y[m] * np.log(p[m]) + (~y[m]) * np.log(1 - p[m])).mean() if m.any() else np.nan
            perr, pc = g["p_end"] - g["point_price"], g["p_end"] - g["p_ref"]
            rows.append({"direction_target": key[0], "symbol": key[1], "session": key[2], "n": len(g), "n_flat": int(flat.sum()),
                         "acc_up_notup": ((g["answer"] == 1) == (g["direction"] == 1)).mean(),
                         "logloss_un": ll(yun, pd.Series(True, index=g.index)), "brier_un": ((p - yun) ** 2).mean(),
                         "logloss_ud": ll(yud, ~flat), "brier_ud": ((p[~flat] - yud[~flat]) ** 2).mean(),
                         "mae_point": perr.abs().mean(), "rmse_point": np.sqrt((perr ** 2).mean()),
                         "mae_pc": pc.abs().mean(), "rmse_pc": np.sqrt((pc ** 2).mean()),
                         "n_answer_flat": int((g["answer"] == 0).sum()),
                         "acc_flat_as_wrong": (hit & ~flat).mean(), "acc_flat_as_correct": (hit | flat).mean(),
                         "acc_flat_excluded": hit[~flat].mean(), "acc_exact_3state": hit.mean(),
                         "n_correct_exact": int(hit.sum())})
        t = pd.DataFrame(rows)
        t.to_csv(out / "live_direction_rules.csv", index=False, encoding="utf-8-sig")
        print("\nทิศทางที่ส่งจริง ตามกติกาวัน flat (ใช้แบบที่อาจารย์กำหนด):")
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
