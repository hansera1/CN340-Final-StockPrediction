"""E11 (NEXT_STEPS รอบที่ 3 ข้อ 9) — PILOT: ResNet18 บน scalogram จากแท่ง 5m ของวันเดียวกัน

- ข้อมูล: yfinance 5m 60 วัน ของ KBANK, ADVANC + หุ้น E10-20 อีก 20 ตัว (หุ้นเพิ่มใช้ฝึกอย่างเดียว วัดผลบน KBANK / ADVANC)
- รอบบ่ายเท่านั้น: scalogram (Morlet) จากแท่ง 5m ตั้งแต่ 10:00 ถึงจุดตัด (แท่งที่จบ ≤ จุดตัด ของวันเดียวกันเท่านั้น)
  3 ช่อง (r, |r|, log volume) · พื้นที่นอก cone of influence (e-folding √2·s จากขอบ) ตั้งเป็น 0
- ResNet18 ขั้น S1 / S2 ตามสเปก E10, 3 seed · walk-forward ฝึกขั้นต่ำ 30 วัน ทดสอบทีละ 5 วัน · ไม่แตะ holdout 10 วันสุดท้าย
- เทียบกับ E0-ud และ E10 ResNet18 S2 (60m) บนแถวทดสอบเดียวกัน

    python experiments/15_e11_pilot.py
→ outputs/experiments/E11/*.csv, results/e11/e11_pilot.csv
"""
import copy

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules, session_times
from forecast.data.io import load_bars
from forecast.evaluation.runner import extra_dataset, load_or_build_dataset, run_variant
from forecast.features.images import _resize, _series, _unit, morlet_cwt

import importlib.util
_spec = importlib.util.spec_from_file_location("s09", ROOT / "experiments" / "09_score_round2.py")
s09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s09)

N_SCALES = 32


def coi_mask(n):
    """True = อยู่ใน cone of influence (ไม่ได้รับผลจากขอบ) ของ scale แบบเดียวกับ morlet_cwt"""
    scales = np.geomspace(1, max(2, n / 2), N_SCALES)
    t = np.arange(n)
    edge = np.minimum(t, n - 1 - t)
    return edge[None, :] >= np.sqrt(2) * scales[:, None]


def same_day_image(day_bars):
    x = _series(day_bars)
    mask = coi_mask(len(day_bars))
    chans = [np.where(mask, _unit(np.log1p(morlet_cwt((c - c.mean()) / (c.std() + 1e-8), N_SCALES))), 0.0)
             for c in x]
    return _resize(np.stack(chans).astype(np.float32))


def images(rows, rules):
    out = np.zeros((len(rows), 3, 224, 224), dtype=np.uint8)
    cache = {}
    for i, (date, sym) in enumerate(rows[["date", "symbol"]].itertuples(index=False)):
        if sym not in cache:
            cache[sym] = load_bars(sym, rules, None, "5m")
        b = cache[sym]
        t = session_times(rules, "afternoon", date)
        d = b[(b["date"] == date) & (b["bar_start"] >= session_times(rules, "morning", date)["start"])
              & (b["bar_end"] <= t["cutoff"])].sort_values("bar_end")
        if len(d) >= 8:
            out[i] = np.round(np.clip(same_day_image(d), 0, 1) * 255).astype(np.uint8)
    return out


def main():
    rules = load_rules()
    r = copy.deepcopy(rules)
    r["evaluation"]["min_train_days"] = 30
    extra_syms = load_experiments()["E10"]["variants"][-1]["extra_symbols"]
    days5 = None
    for sym in rules["symbols"]:
        d = set(load_bars(sym, rules, None, "5m")["date"].unique())
        days5 = d if days5 is None else days5 & d
    ds = load_or_build_dataset(rules)
    ds = ds[(ds["session"] == "afternoon") & ds["date"].isin(days5)]
    ext = extra_dataset(rules, extra_syms)
    ext = ext[(ext["session"] == "afternoon") & ext["date"].isin(ds["date"].unique())]
    both = pd.concat([ds, ext], ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"E11: {both['date'].nunique()} วัน ({both['date'].min().date()} – {both['date'].max().date()}), "
          f"{both['symbol'].nunique()} หุ้น, {len(both)} แถว")
    img = images(both, rules)
    out = OUTPUT_DIR / "experiments" / "E11"
    out.mkdir(parents=True, exist_ok=True)
    preds = {}
    for stage in ["s1", "s2"]:
        v = {"name": f"resnet18_5m_{stage}", "model": "finetune", "group_by": ["session"],
             "params": {"arch": "resnet18", "stage": stage}}
        p = run_variant(both, v, r, seqs=img, test_symbols=rules["symbols"])
        p.to_csv(out / f"{v['name']}_predictions.csv", index=False)
        preds[f"E11 ResNet18 S{stage[1]} (5m)"] = p
        print(f"  {v['name']}: {len(p)} แถวทดสอบ", flush=True)
    e0 = run_variant(ds.reset_index(drop=True), {"name": "baseline_ud", "model": "baseline",
                                                 "params": {"target": "updown"}}, r)
    e0.to_csv(out / "baseline_ud_predictions.csv", index=False)
    preds["E0-ud"] = e0
    tz = rules["timezone"]
    preds["E10 ResNet18 S2 (60m)"] = s09.load("E10", "resnet18_s2", tz)
    keys = None
    for p in preds.values():
        k = set(map(tuple, p[s09.KEYS].astype(str).to_numpy()))
        keys = k if keys is None else keys & k
    preds = {n: p[p[s09.KEYS].astype(str).apply(tuple, axis=1).isin(keys)].reset_index(drop=True)
             for n, p in preds.items()}
    q, ev = rules["quantiles"], rules["evaluation"]
    rows = []
    for n, p in preds.items():
        s = s09.scores(p, q)
        rows.append({"model": n, "n_days": p["date"].nunique(), "n_nonflat": s["n_nonflat"], "acc_nf": s["acc_nf"],
                     "logloss_ud": s["logloss_ud"],
                     **(s09.boot(p, preds["E0-ud"], q, ev, None) if n != "E0-ud" else {})})
    t = pd.DataFrame(rows)
    t.insert(0, "note", "PILOT: รอบบ่าย, yfinance 5m 60 วัน, ฝึกขั้นต่ำ 30 วัน — ไม่ใช้เลือกโมเดล")
    (ROOT / "results" / "e11").mkdir(parents=True, exist_ok=True)
    t.to_csv(ROOT / "results" / "e11" / "e11_pilot.csv", index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
