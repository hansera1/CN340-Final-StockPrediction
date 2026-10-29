"""รันการทดลองตาม configs/experiments.yaml ด้วย walk-forward แล้วเทียบกับ variant แรกด้วย block bootstrap

    python scripts/03_run_experiment.py E1 [E2 ...] [--raw-dir ...] [--holdout]
    → outputs/experiments/<E>/summary.csv, bootstrap.csv, <variant>_predictions.csv

--holdout ใช้หลังล็อกวิธี (5 พ.ย.) เท่านั้น: ประเมินช่วงท้ายที่กันไว้ครั้งเดียว
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, load_experiments, load_rules
from forecast.evaluation.bootstrap import compare
from forecast.evaluation.metrics import summarize
from forecast.evaluation.runner import load_or_build_dataset, run_variant, sequences_for


def run(code, rules, raw_dir, holdout):
    exp = load_experiments()[code]
    out = OUTPUT_DIR / "experiments" / (code + ("_holdout" if holdout else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n### {code}: {exp['question']}  (ไอเดียเดิม: {exp['legacy']})")

    q = rules["quantiles"]
    ev = rules["evaluation"]
    preds, summaries = {}, []
    for v in exp["variants"]:
        ds = load_or_build_dataset(rules, v.get("feature_minutes", 1), v.get("spectral", False), raw_dir)
        seqs = sequences_for(ds, rules, raw_dir) if v["model"] == "cnn1d" else None
        p = run_variant(ds, v, rules, seqs=seqs, holdout_eval=holdout)
        p.to_csv(out / f"{v['name']}_predictions.csv", index=False)
        s = summarize(p, q)
        s.insert(0, "variant", v["name"])
        summaries.append(s)
        preds[v["name"]] = p

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    show = ["variant", "group", "n", "n_flat", "acc_flat_as_wrong", "acc_flat_excluded",
            "balanced_accuracy", "logloss", "brier", "mae_baht", "mae_naive_baht", "coverage50", "coverage80"]
    print(summary[show].round(4).to_string(index=False))

    names = list(preds)
    if len(names) > 1:
        rows = []
        for other in names[1:]:
            c = compare(preds[other], preds[names[0]], q, block_days=ev["block_days"], n_boot=ev["n_boot"])
            c.insert(0, "comparison", f"{other} − {names[0]}")
            rows.append(c)
        boot = pd.concat(rows, ignore_index=True)
        boot.to_csv(out / "bootstrap.csv", index=False)
        print("\nblock bootstrap (diff < 0 = variant ใหม่ดีกว่า):")
        print(boot.round(5).to_string(index=False))
    print(f"→ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    rules = load_rules()
    for code in args.experiments:
        run(code, rules, args.raw_dir, args.holdout)


if __name__ == "__main__":
    main()
