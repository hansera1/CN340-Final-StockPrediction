"""E13 — ชั้น 2 ปรับทิศทางรอบบ่ายด้วยแท่ง 5m (สเปกและเกณฑ์ล็อกใน results/e13_preregistration.md)

    python experiments/21_e13_stage2.py e2          # ตรวจแหล่งข้อมูล: Settrade 1m → 5m เทียบ yfinance 5m (รันซ้ำได้)
    python experiments/21_e13_stage2.py backtest    # รันครั้งเดียว หลังตั้งจุดตัดรอบบ่ายจากการวัดความหน่วง (A3)
→ results/e13/

จุดตัด = rules.yaml → sessions.afternoon.cutoff ค่าเดียว · backtest ไม่ยอมรันถ้ายังไม่มี results/latency_2026-11-02.md
หรือถ้ามีผล backtest แล้ว (ห้ามรันซ้ำ)
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import ROOT, load_rules, raw_source_dir, session_times
from forecast.data.io import aggregate_bars, load_bars
from forecast.evaluation.metrics import tick_size
from forecast.features.fine import F5, f5_features
from forecast.live.stage2_live import f5_history, p1_backtest
from forecast.models.stage2 import fit_for_day

OUT = ROOT / "results" / "e13"
EPS = 1e-4


def run_e2(rules):
    """Settrade 1m รวมเป็น 5m เทียบ yfinance 5m: OHLC ทุกแท่งภาคเช้า (ต่าง ≤ 1 ช่อง) และเครื่องหมาย f5_* ณ จุดตัด"""
    bars, f5s = [], []
    for sym in rules["symbols"]:
        st = load_bars(sym, rules, raw_source_dir(rules, "settrade"), "1m")
        yf = load_bars(sym, rules, raw_source_dir(rules, "yfinance"), "5m")
        for date in sorted(set(st["date"]) & set(yf["date"])):
            a = aggregate_bars(st[st["date"] == date], 5, rules)
            y = yf[yf["date"] == date]
            end = session_times(rules, "morning", date)["end"]
            m = a[a["bar_start"] < end].merge(y[y["bar_start"] < end], on="bar_start", suffixes=("_st", "_yf"))
            for c in ["open", "high", "low", "close"]:
                ticks = (m[f"{c}_st"] - m[f"{c}_yf"]).abs() / tick_size(m[f"{c}_yf"])
                bars.append(pd.DataFrame({"symbol": sym, "date": date, "bar_start": m["bar_start"], "field": c,
                                          "diff_ticks": ticks}))
            fs, fy = f5_features(a, date, rules), f5_features(y, date, rules)
            same = {"f5_r_last15": np.sign(fs["f5_r_last15"]) == np.sign(fy["f5_r_last15"]),
                    "f5_close_vs_twap": np.sign(fs["f5_close_vs_twap"]) == np.sign(fy["f5_close_vs_twap"]),
                    "f5_range_pos": np.sign(fs["f5_range_pos"] - 0.5) == np.sign(fy["f5_range_pos"] - 0.5)}
            f5s.append({"symbol": sym, "date": date.date(), **{f"{k}_st": fs[k] for k in F5},
                        **{f"{k}_yf": fy[k] for k in F5}, **{f"{k}_sign_ok": v for k, v in same.items()},
                        "bars_compared": len(m)})
    b, f = pd.concat(bars), pd.DataFrame(f5s)
    share = (b.groupby(["symbol", "date", "bar_start"])["diff_ticks"].max() <= 1).mean()
    signs = f[[f"{k}_sign_ok" for k in F5]].all(axis=1).all()
    passed = bool(share >= 0.95 and signs)
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT / "e2_f5_compare.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{"cutoff": rules["sessions"]["afternoon"]["cutoff"], "n_days": f["date"].nunique(),
                             "n_symbol_days": len(f), "n_bars": int(b.groupby(["symbol", "date", "bar_start"]).ngroups),
                             "share_bars_within_1_tick": share, "max_diff_ticks": b["diff_ticks"].max(),
                             "f5_signs_all_match": bool(signs), "pass": passed}])
    summary.to_csv(OUT / "e2_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.T.to_string(header=False))
    print(f.to_string(index=False))
    return passed


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(a, b, dates, n_boot, seed=0):
    """ค่าเฉลี่ยของ a − b + 95% CI · block bootstrap รายวันแบบจับคู่ (ทุกแถวของวันเดียวกันไปด้วยกัน)"""
    d = pd.DataFrame({"v": np.asarray(a, float) - np.asarray(b, float), "d": np.asarray(dates)}).dropna()
    g = d.groupby("d")["v"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def run_backtest(rules, force=False):
    cfg = rules["live"]["stage2"]
    if not (ROOT / "results" / "latency_2026-11-02.md").exists() and not force:
        raise SystemExit("ยังไม่มี results/latency_2026-11-02.md — ตั้งจุดตัดรอบบ่ายจากการวัดความหน่วงก่อน (A3)")
    if (OUT / "backtest_rows.csv").exists() and not force:
        raise SystemExit("มีผล backtest แล้ว — ตามการลงทะเบียน backtest รันครั้งเดียว")
    n_boot = rules["evaluation"]["n_boot"]
    t = p1_backtest(rules)
    f = pd.concat([f5_history(rules, s, placebo=True) for s in rules["symbols"]])
    t = t.merge(f, on=["date", "symbol"], how="inner")
    plc = t.drop(columns=F5).rename(columns={f"plc_{k}": k for k in F5})
    rows, betas = [], []
    for date in sorted(t["date"].unique()):
        day = t[(t["date"] == date)].dropna(subset=["p1", *F5])
        if day.empty:
            continue
        m, info = fit_for_day(t, date, cfg["train_days"], cfg["l2"], cfg["min_nonflat"])
        if info["n_days"] < cfg["train_days"]:
            continue                                       # วันฝึกยังไม่ครบ 20 วัน → ไม่ใช่วันทดสอบ
        mp, _ = fit_for_day(plc, date, cfg["train_days"], cfg["l2"], cfg["min_nonflat"])
        dp = plc[plc["date"] == date].set_index("symbol").loc[day["symbol"]]
        p12 = m.predict(day[F5], day["p1"]) if m else day["p1"].to_numpy()
        ok_p = dp[F5].notna().all(axis=1).to_numpy()
        pp = day["p1"].to_numpy().copy()
        if mp is not None and ok_p.any():
            pp[ok_p] = mp.predict(dp.loc[ok_p, F5], dp.loc[ok_p, "p1"])
        rows.append(day.assign(p12=p12, p12_placebo=pp, stage2_used=m is not None, n_train_nonflat=info["n_nonflat"]))
        betas.append({"date": date.date(), **info, **({f"beta_{k}": b for k, b in zip(F5, m.beta_)} if m else {})})
    r = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "backtest_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(betas).to_csv(OUT / "betas.csv", index=False, encoding="utf-8-sig")

    nf = r[r["direction"].notna() & (r["direction"] != 0)].copy()
    y = (nf["direction"] == 1).to_numpy(float)
    ll = {k: logloss(nf[c].to_numpy(), y) for k, c in [("S1", "p1"), ("S12", "p12"), ("S12-placebo", "p12_placebo"),
                                                       ("E0-ud", "p_e0")]}
    hit = {k: ((nf[c] >= 0.5).to_numpy() == (y == 1)).astype(float)
           for k, c in [("S1", "p1"), ("S12", "p12"), ("S12-placebo", "p12_placebo"), ("E0-ud", "p_e0")]}
    lab = r[r["direction"].notna()]
    fw = {k: (((lab[c] >= 0.5) & (lab["direction"] == 1)) | ((lab[c] < 0.5) & (lab["direction"] == -1))).mean()
          for k, c in [("S1", "p1"), ("S12", "p12"), ("S12-placebo", "p12_placebo"), ("E0-ud", "p_e0")]}
    flip = (r["p12"] >= 0.5) != (r["p1"] >= 0.5)
    flip_nf = (nf["p12"] >= 0.5) != (nf["p1"] >= 0.5)
    summ = []
    for k in ll:
        row = {"model": k, "n_days": r["date"].nunique(), "n_rounds": len(r), "n_nonflat": len(nf),
               "logloss_ud": ll[k].mean(), "acc_nf": hit[k].mean(), "acc_fw": fw[k]}
        if k != "S1":
            row.update(dict(zip(["d_logloss_vs_S1", "d_logloss_lo", "d_logloss_hi"],
                                boot_diff(ll[k], ll["S1"], nf["date"], n_boot))))
            row.update(dict(zip(["d_acc_vs_S1", "d_acc_lo", "d_acc_hi"],
                                boot_diff(hit[k], hit["S1"], nf["date"], n_boot))))
        summ.append(row)
    summ = pd.DataFrame(summ)
    per = []
    for sym, g in nf.groupby("symbol"):
        ix = nf.index.get_indexer(g.index)
        a = boot_diff(ll["S12"][ix], ll["S1"][ix], g["date"], n_boot)
        b = boot_diff(hit["S12"][ix], hit["S1"][ix], g["date"], n_boot)
        per.append({"symbol": sym, "n_nonflat": len(g), "d_logloss": a[0], "d_logloss_lo": a[1], "d_logloss_hi": a[2],
                    "d_acc": b[0], "d_acc_lo": b[1], "d_acc_hi": b[2]})
    per = pd.DataFrame(per)
    vs_plc = boot_diff(ll["S12"], ll["S12-placebo"], nf["date"], n_boot)
    e2 = pd.read_csv(OUT / "e2_summary.csv")["pass"].iat[0] if (OUT / "e2_summary.csv").exists() else False
    s12 = summ.set_index("model").loc["S12"]
    crit = pd.DataFrame([
        {"criterion": "1 Δ log loss < 0 และขอบบน CI < 0", "value": f"{s12.d_logloss_vs_S1:+.4f} [{s12.d_logloss_lo:+.4f}, {s12.d_logloss_hi:+.4f}]",
         "pass": bool(s12.d_logloss_vs_S1 < 0 and s12.d_logloss_hi < 0)},
        {"criterion": "2 Δ ความแม่นไม่นับ flat ≥ 0", "value": f"{100 * s12.d_acc_vs_S1:+.1f} จุด", "pass": bool(s12.d_acc_vs_S1 >= 0)},
        {"criterion": "3 ไม่มีหุ้นที่แย่กว่าอย่างมีนัยสำคัญ",
         "value": "; ".join(f"{x.symbol}: ll {x.d_logloss:+.4f} [{x.d_logloss_lo:+.4f}, {x.d_logloss_hi:+.4f}], "
                            f"acc {100 * x.d_acc:+.1f} [{100 * x.d_acc_lo:+.1f}, {100 * x.d_acc_hi:+.1f}]" for x in per.itertuples()),
         "pass": bool(((per["d_logloss_lo"] <= 0) & (per["d_acc_hi"] >= 0)).all())},
        {"criterion": "4 S12 ดีกว่า placebo (Δ log loss < 0)", "value": f"{vs_plc[0]:+.4f} [{vs_plc[1]:+.4f}, {vs_plc[2]:+.4f}]",
         "pass": bool(vs_plc[0] < 0)},
        {"criterion": "5 ผ่าน E-2", "value": str(e2), "pass": bool(e2)},
        {"criterion": "6 พลิกทิศทาง ≤ 20%", "value": f"{100 * flip.mean():.1f}% ของ {len(r)} รอบ", "pass": bool(flip.mean() <= 0.20)},
    ])
    extra = pd.DataFrame([{"cutoff": rules["sessions"]["afternoon"]["cutoff"], "n_days": r["date"].nunique(),
                           "first_day": r["date"].min().date(), "last_day": r["date"].max().date(),
                           "n_rounds": len(r), "n_nonflat": len(nf), "share_days_stage2_used": r["stage2_used"].mean(),
                           "flip_share": flip.mean(), "acc_on_flipped_nonflat_S12": hit["S12"][flip_nf.to_numpy()].mean()
                           if flip_nf.any() else np.nan, "n_flipped_nonflat": int(flip_nf.sum()),
                           "mean_abs_p12_minus_p1": (r["p12"] - r["p1"]).abs().mean()}])
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    per.to_csv(OUT / "per_symbol.csv", index=False, encoding="utf-8-sig")
    crit.to_csv(OUT / "criteria.csv", index=False, encoding="utf-8-sig")
    extra.to_csv(OUT / "describe.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))
    print(per.round(4).to_string(index=False))
    print(extra.T.to_string(header=False))
    print(crit.to_string(index=False))
    print("→ ผ่านทุกข้อ: S12 ส่งจริง" if crit["pass"].all() else "→ ไม่ผ่าน: ส่ง S1 และรัน S12 แบบเงา")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["e2", "backtest"])
    ap.add_argument("--force", action="store_true", help="ใช้ใน test เท่านั้น — ข้ามการกันรันซ้ำ")
    args = ap.parse_args()
    rules = load_rules()
    print(f"จุดตัดรอบบ่าย = {rules['sessions']['afternoon']['cutoff']}")
    run_e2(rules) if args.what == "e2" else run_backtest(rules, args.force)


if __name__ == "__main__":
    main()
