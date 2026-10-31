"""รันการทดลองตาม configs/experiments.yaml ด้วย walk-forward แล้วเทียบกับ variant แรกด้วย block bootstrap

    python experiments/03_run_experiment.py E1 [E2 ...] [--raw-dir ...] [--holdout]
    → outputs/experiments/<E>/summary.csv, bootstrap.csv, <variant>_predictions.csv

--holdout ใช้หลังล็อกวิธี (5 พ.ย.) เท่านั้น: ประเมินช่วงท้ายที่กันไว้ครั้งเดียว
"""
import argparse
import time
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_experiments, load_rules
from forecast.evaluation.bootstrap import compare
from forecast.evaluation.metrics import row_losses, summarize
from forecast.evaluation.runner import (embeddings_for, experiment_rules, extra_dataset, images_for,
                                       load_or_build_dataset,
                                       overlap_dates, run_variant, sequences_for, variant_interval)

KEEP_COLS = ["date", "symbol", "session", "p_ref", "p_end", "z", "r", "direction", "up"]
METRICS = ("logloss", "logloss_ud", "err_nf", "err_fw", "brier", "abs_err_baht", "pinball")
PILOT_NOTE = "⚠️ PILOT — ข้อมูลน้อย ({days} วัน, {interval}) ใช้ดูทิศทางเท่านั้น ไม่ใช้ตัดสินเลือกวิธีหลัก"


def load_predictions(code, name, rules, holdout):
    path = OUTPUT_DIR / "experiments" / (code + ("_holdout" if holdout else "")) / f"{name}_predictions.csv"
    if not path.exists():
        raise SystemExit(f"ยังไม่มี {path} — รัน {code} ก่อน")
    p = pd.read_csv(path)
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(rules["timezone"])
    return p


def report_seeds(name, p, quantiles):
    """โมเดลหลาย seed: log loss ราย seed (ค่าเฉลี่ย + ช่วง) — ไม่เลือก seed ที่ดีที่สุด"""
    cols = [c for c in p if c.startswith("p_up_seed")]
    if not cols:
        return
    ll = [row_losses(p.assign(p_up=p[c]), quantiles)["logloss"].mean() for c in cols]
    print(f"  {name}: log loss ราย seed เฉลี่ย {np.mean(ll):.4f} ช่วง {min(ll):.4f}–{max(ll):.4f} "
          f"({len(cols)} seeds); ค่าเฉลี่ย p ของทุก seed = แถว {name} ในตาราง")


def run(code, rules, raw_dir, holdout, only=None, resume=False):
    exp = load_experiments()[code]
    rules = experiment_rules(rules, exp)
    out = OUTPUT_DIR / "experiments" / (code + ("_holdout" if holdout else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n### {code}: {exp['question']}  (ไอเดียเดิม: {exp['legacy']})")

    q = rules["quantiles"]
    ev = rules["evaluation"]
    keep = overlap_dates(rules, exp["overlap_intervals"], raw_dir) if exp.get("overlap_intervals") else None
    if keep:
        print(f"ใช้เฉพาะวันที่มีข้อมูลครบทุกความละเอียด {exp['overlap_intervals']}: "
              f"{len(keep)} วัน ({keep[0].date()} – {keep[-1].date()})")
    preds, summaries, notes = {}, [], []
    for v in exp["variants"]:
        if only and v["name"] not in only:
            continue
        interval = variant_interval(rules, exp, v)
        done = out / f"{v['name']}_predictions.csv"
        t_v = time.perf_counter()
        ds = load_or_build_dataset(rules, v.get("feature_minutes"), v.get("spectral", False), raw_dir,
                                   interval=interval)
        if keep is not None:
            ds = ds[ds["date"].isin(keep)].reset_index(drop=True)
        if v.get("ensemble"):     # เฉลี่ย P(ขึ้น) ของ variant ก่อนหน้า (แถวเดียวกัน) ช่วงราคาใช้ของตัวแรก
            parts = [preds[n] for n in v["ensemble"]]
            p = parts[0][KEEP_COLS + [c for c in parts[0] if c == "p_up" or c.startswith("q")]].copy()
            p["p_up"] = np.mean([x["p_up"].to_numpy() for x in parts], axis=0)
        elif v.get("reuse"):        # ใช้ผลของการทดลองก่อนหน้า (ข้อมูล + walk-forward เดียวกัน) ไม่ฝึกซ้ำ
            p = load_predictions(v["reuse"], v["name"], rules, holdout)
            p = p[p["date"].isin(ds["date"])].reset_index(drop=True)
        elif resume and done.exists():   # รันแล้ว (เช่น E10 ที่ใช้เวลานาน) → ใช้ผลเดิม
            p = load_predictions(code + ("_holdout" if holdout else ""), v["name"], rules, False)
        elif v.get("extra_symbols"):    # E10-20: หุ้นอื่นเป็นข้อมูลฝึกเพิ่ม วัดผลบนหุ้นเป้าหมายเท่านั้น
            ext = extra_dataset(rules, v["extra_symbols"], raw_dir, interval)
            ext = ext[ext["date"].isin(ds["date"].unique())]
            both = pd.concat([ds, ext], ignore_index=True)
            print(f"  {v['name']}: ฝึกด้วย {both['symbol'].nunique()} หุ้น ({len(both)} แถว), ทดสอบ {rules['symbols']}")
            seqs = images_for(both, rules, v["image"], raw_dir, interval, tag=f"_x{len(v['extra_symbols'])}")
            p = run_variant(both, v, rules, seqs=seqs, holdout_eval=holdout, test_symbols=rules["symbols"])
            del seqs
            from forecast.models import finetune
            pd.DataFrame(finetune.CURVES).to_csv(out / f"{v['name']}_curves.csv", index=False)
            finetune.CURVES.clear()
        else:
            if v["model"] == "finetune":
                seqs = images_for(ds, rules, v["image"], raw_dir, interval)
            elif v["model"] == "cnn1d":
                seqs = sequences_for(ds, rules, raw_dir, interval)
            elif v.get("image"):
                seqs = embeddings_for(ds, rules, v["image"], raw_dir, interval)
            else:
                seqs = None
            p = run_variant(ds, v, rules, seqs=seqs, holdout_eval=holdout)
            if v["model"] == "finetune":   # learning curve ราย fold / seed / epoch
                from forecast.models import finetune
                pd.DataFrame(finetune.CURVES).to_csv(out / f"{v['name']}_curves.csv", index=False)
                finetune.CURVES.clear()
        p.to_csv(out / f"{v['name']}_predictions.csv", index=False)
        print(f"  {v['name']}: เสร็จ {time.strftime('%H:%M:%S')} ({time.perf_counter() - t_v:.0f} วินาที)", flush=True)
        if exp.get("custom_scoring"):
            continue
        report_seeds(v["name"], p, q)
        s = summarize(p, q)
        s.insert(0, "variant", v["name"])
        s.insert(1, "interval", interval)
        s.insert(2, "pilot", bool(exp.get("pilot")))
        summaries.append(s)
        preds[v["name"]] = p
        if exp.get("pilot"):
            notes.append(PILOT_NOTE.format(days=ds["date"].nunique(), interval=interval))

    if exp.get("custom_scoring"):
        print(f"→ {out} (คิดคะแนนด้วยสคริปต์เฉพาะของการทดลองนี้)")
        return
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    if notes:
        text = "\n".join(dict.fromkeys(notes))
        (out / "PILOT.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
    show = ["variant", "group", "n", "n_flat", "acc_flat_as_wrong", "acc_flat_excluded",
            "balanced_accuracy", "logloss", "logloss_ud", "brier", "mae_baht", "mae_naive_baht", "coverage50", "coverage80"]
    print(summary[show].round(4).to_string(index=False))

    names = list(preds)
    if len(names) > 1:
        rows = []
        for other in names[1:]:
            c = compare(preds[other], preds[names[0]], q, METRICS, block_days=ev["block_days"], n_boot=ev["n_boot"])
            c.insert(0, "comparison", f"{other} − {names[0]}")
            c.insert(1, "pilot", bool(exp.get("pilot")))
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
    ap.add_argument("--only", help="รันเฉพาะ variant เหล่านี้ (คั่นด้วย ,)")
    ap.add_argument("--resume", action="store_true", help="variant ที่มี predictions แล้วไม่ฝึกซ้ำ")
    args = ap.parse_args()
    rules = load_rules()
    only = set(args.only.split(",")) if args.only else None
    for code in args.experiments:
        run(code, rules, args.raw_dir, args.holdout, only, args.resume)


if __name__ == "__main__":
    main()
