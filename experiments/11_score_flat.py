"""คิดคะแนน E9-FLAT (docs/NEXT_STEPS.md ข้อ 2) จาก predictions ที่มีอยู่ — ไม่ฝึกใหม่ ไม่แตะ holdout

    python experiments/11_score_flat.py
    → outputs/reports/round2/flat_*.csv

1. Model A (A-dir: ปลาย = ต้นช่วง, A-price: ปลาย = P(c)) — log loss / Brier เทียบ baseline สัดส่วนวันนิ่งในอดีต
   + calibration แยกหุ้น/รอบ (CI ของสัดส่วนจริงต่อช่วงความน่าจะเป็นจาก block bootstrap รายวัน)
2. ทิศทาง 3 คำตอบ: P(ขึ้น) = (1 − f)·p, P(ลง) = (1 − f)·(1 − p), P(นิ่ง) = f → เลือกค่าสูงสุด
   p = E9 ตัวที่ดีที่สุด (ความแม่นไม่นับ flat), f = A-dir ตัวที่ log loss ต่ำสุด
   นับ "เท่าเดิม" เป็นคำตอบที่ถูกได้ เทียบ (ก) E0, (ข) E9 ขึ้น/ลงอย่างเดียว, (ค) ทาย "เท่าเดิม" ทุกรอบ
3. ราคากฎค่ามัธยฐาน: +1 ช่องถ้า (1 − f_price)·p_price > 0.5, −1 ช่องถ้า (1 − f_price)·(1 − p_price) > 0.5
   นอกนั้น P(c) — เทียบ MAE กับ z = 0 (เกณฑ์ 0.5 ตามแผน ไม่ปรับ)
ทุกตัวคิดบนแถวทดสอบชุดเดียวกับ E9 (walk-forward เดียวกัน)
"""
import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules
from forecast.evaluation.metrics import tick_size
from forecast.models.e9 import same_as_ref

EXP = OUTPUT_DIR / "experiments"
OUT = OUTPUT_DIR / "reports" / "round2"
KEYS = ["date", "symbol", "session"]
GROUPS = [None, ("KBANK", "morning"), ("KBANK", "afternoon"), ("ADVANC", "morning"), ("ADVANC", "afternoon")]
EPS = 1e-4


def load(exp, variant, tz):
    p = pd.read_csv(EXP / exp / f"{variant}_predictions.csv")
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def gname(g):
    return "ALL" if g is None else f"{g[0]}/{g[1]}"


def subset(d, g):
    return d if g is None else d[(d["symbol"] == g[0]) & (d["session"] == g[1])]


def boot_mean(values, dates, n_boot, seed=0):
    """ค่าเฉลี่ยต่อแถว + 95% CI จาก block bootstrap รายวัน (บล็อก 1 วัน; ทุกแถวของวันเดียวกันไปด้วยกัน)"""
    df = pd.DataFrame({"v": np.asarray(values, float), "d": np.asarray(dates)}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    daily = df.groupby("d")["v"].agg(["sum", "count"])
    s, c = daily["sum"].to_numpy(), daily["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score_model_a(preds, label, n_boot):
    """preds: {ชื่อ: DataFrame} ของเป้าหมายเดียว, label: Series 0/1 (NaN = ไม่มีผล) ตาม index ของแถวร่วม"""
    rows, calib = [], []
    base = preds["base"]
    for name, p in preds.items():
        y = label.to_numpy(float)
        ll, br = logloss(p["p_up"].to_numpy(), y), (p["p_up"].to_numpy() - y) ** 2
        llb = logloss(base["p_up"].to_numpy(), y)
        brb = (base["p_up"].to_numpy() - y) ** 2
        for g in GROUPS:
            m = subset(p, g).index
            d = p.loc[m, "date"]
            r = {"model": name, "group": gname(g), "n": int(np.isfinite(y[m]).sum()), "flat_rate": np.nanmean(y[m]),
                 "mean_p": p.loc[m, "p_up"].mean(), "logloss": np.nanmean(ll[m]), "brier": np.nanmean(br[m])}
            if name != "base":
                for k, a, b in [("logloss", ll, llb), ("brier", br, brb)]:
                    mean, lo, hi = boot_mean(a[m] - b[m], d, n_boot)
                    r.update({f"d_{k}": mean, f"d_{k}_lo": lo, f"d_{k}_hi": hi})
            rows.append(r)
            if g is not None:   # calibration: 5 ช่วงตาม quantile ของความน่าจะเป็นที่ทาย
                pp = p.loc[m, "p_up"]
                ok = np.isfinite(y[m])
                try:
                    bins = pd.qcut(pp[ok], 5, labels=False, duplicates="drop")
                except ValueError:
                    continue
                for b in sorted(bins.unique()):
                    mm = bins.index[bins == b]
                    obs, lo, hi = boot_mean(label.loc[mm], p.loc[mm, "date"], n_boot)
                    calib.append({"model": name, "group": gname(g), "bin": int(b), "n": len(mm),
                                  "mean_pred": pp.loc[mm].mean(), "observed": obs, "obs_lo": lo, "obs_hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(calib)


def compare_rows(correct, ref_correct, dates, n_boot):
    return boot_mean(np.asarray(correct, float) - np.asarray(ref_correct, float), dates, n_boot)


def main():
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    OUT.mkdir(parents=True, exist_ok=True)

    # แถวทดสอบชุดเดียวกับ E9
    e9 = {v: load("E9", v, tz) for v in ["lr", "lrx", "gbm", "ens", "baseline_ud"]}
    keys = e9["lr"][KEYS]
    align = lambda p: keys.merge(p, on=KEYS, how="left")
    e9 = {k: align(v) for k, v in e9.items()}
    flat = {v: align(load("E9FLAT", v, tz)) for v in
            ["adir_base", "adir_lr", "adir_gbm", "aprice_base", "aprice_lr", "aprice_gbm", "pprice_lr"]}
    e0 = align(load("E0", "baseline", tz))
    d = e9["lr"]
    labeled = d["direction"].notna() & d["p_end"].notna()
    keep = labeled.to_numpy()
    e9 = {k: v[keep].reset_index(drop=True) for k, v in e9.items()}
    flat = {k: v[keep].reset_index(drop=True) for k, v in flat.items()}
    e0 = e0[keep].reset_index(drop=True)
    d = e9["lr"]
    dates = d["date"]
    print(f"แถวทดสอบร่วมกับ E9: {len(d)} รอบ ({dates.min().date()} – {dates.max().date()})")
    for k, v in flat.items():
        miss = v["p_up"].isna().sum()
        if miss:
            print(f"  ⚠ {k}: ไม่มีผล {miss} แถว")

    # ---------- 1. Model A
    y_dir = (d["direction"] == 0).astype(float)
    ok, same = same_as_ref(d)
    y_price = pd.Series(np.where(ok, same, np.nan), index=d.index)
    a_dir, cal_dir = score_model_a({"base": flat["adir_base"], "lr": flat["adir_lr"], "gbm": flat["adir_gbm"]},
                                   y_dir, n_boot)
    a_price, cal_price = score_model_a({"base": flat["aprice_base"], "lr": flat["aprice_lr"],
                                        "gbm": flat["aprice_gbm"]}, y_price, n_boot)
    a_dir.insert(0, "target", "A-dir")
    a_price.insert(0, "target", "A-price")
    cal_dir.insert(0, "target", "A-dir")
    cal_price.insert(0, "target", "A-price")
    pd.concat([a_dir, a_price]).to_csv(OUT / "flat_model_a.csv", index=False)
    pd.concat([cal_dir, cal_price]).to_csv(OUT / "flat_calibration.csv", index=False)

    def best(tab):
        t = tab[(tab["group"] == "ALL") & (tab["model"] != "base")]
        return t.sort_values("logloss")["model"].iat[0]

    best_dir, best_price = best(a_dir), best(a_price)

    # ---------- 2. ทิศทาง 3 คำตอบ
    nf = d["direction"] != 0
    acc_nf = {k: ((v["p_up"] >= 0.5) == (v["direction"] == 1))[nf].mean() for k, v in e9.items() if k != "baseline_ud"}
    best_e9 = max(acc_nf, key=acc_nf.get)
    p = e9[best_e9]["p_up"].to_numpy()
    direction = d["direction"].to_numpy()
    two = lambda pu: np.where(pu >= 0.5, 1, -1)

    def three(f):
        probs = np.column_stack([(1 - f) * p, f, (1 - f) * (1 - p)])        # ขึ้น, นิ่ง, ลง
        return np.array([1, 0, -1])[probs.argmax(1)]

    answers = {
        f"3 คำตอบ (E9-{best_e9.upper()} + A-dir {best_dir.upper()})": three(flat[f"adir_{best_dir}"]["p_up"].to_numpy()),
        "3 คำตอบ (E9 + A-dir baseline)": three(flat["adir_base"]["p_up"].to_numpy()),
        "(ก) E0 รอบแรก (ขึ้น/ลง)": two(e0["p_up"].to_numpy()),
        "(ก) E0-ud (ขึ้น/ลง)": two(e9["baseline_ud"]["p_up"].to_numpy()),
        f"(ข) E9-{best_e9.upper()} ขึ้น/ลงอย่างเดียว": two(p),
        "(ค) ทาย \"เท่าเดิม\" ทุกรอบ": np.zeros(len(d)),
    }
    main3 = next(iter(answers))
    rows = []
    for name, a in answers.items():
        corr = (a == direction).astype(float)
        for g in GROUPS:
            m = subset(d, g).index
            r = {"answer": name, "group": gname(g), "n": len(m), "acc3": corr[m].mean(),
                 "share_up": (a[m] == 1).mean(), "share_flat": (a[m] == 0).mean(), "share_down": (a[m] == -1).mean()}
            if name != main3:
                ref = (answers[main3] == direction).astype(float)
                r.update(dict(zip(["d_main_minus_this", "d_lo", "d_hi"],
                                  compare_rows(ref[m], corr[m], dates[m], n_boot))))
            rows.append(r)
    three_tab = pd.DataFrame(rows)
    three_tab.to_csv(OUT / "flat_three_way.csv", index=False)

    # ---------- 3. ราคากฎค่ามัธยฐาน
    fp = flat[f"aprice_{best_price}"]["p_up"].to_numpy()
    pp = flat["pprice_lr"]["p_up"].to_numpy()
    up, dn = (1 - fp) * pp, (1 - fp) * (1 - pp)
    tick = tick_size(d["p_ref"]).astype(float)
    shift = np.where(up > 0.5, 1, np.where(dn > 0.5, -1, 0))
    err0 = (d["p_end"] - d["p_ref"]).abs().to_numpy()
    err1 = (d["p_end"] - (d["p_ref"] + shift * tick)).abs().to_numpy()
    rows = []
    for g in GROUPS:
        m = subset(d, g).index
        dm, lo, hi = boot_mean(err1[m] - err0[m], dates[m], n_boot)
        rows.append({"group": gname(g), "n": len(m), "n_shift_up": int((shift[m] == 1).sum()),
                     "n_shift_down": int((shift[m] == -1).sum()),
                     "max_p_up_side": up[m].max(), "max_p_down_side": dn[m].max(),
                     "mae_z0_baht": err0[m].mean(), "mae_rule_baht": err1[m].mean(),
                     "mae_z0_ticks": (err0[m] / tick[m]).mean(), "mae_rule_ticks": (err1[m] / tick[m]).mean(),
                     "d_mae_baht": dm, "d_mae_lo": lo, "d_mae_hi": hi})
    price_tab = pd.DataFrame(rows)
    price_tab.to_csv(OUT / "flat_price_rule.csv", index=False)

    # คุณภาพของ p_price (ราคาปลาย > P(c) บนรอบที่ราคาปลาย ≠ P(c)) เทียบสัดส่วนในอดีตแบบง่าย
    m = ok & ~same
    yp = (d["p_end"] > d["p_ref"]).to_numpy(float)[m]
    pps = pd.DataFrame([{"n": int(m.sum()), "up_rate": yp.mean(), "acc": ((pp[m] >= 0.5) == (yp == 1)).mean(),
                         "logloss": logloss(pp[m], yp).mean(), "share_p_gt_0_5": (pp[m] >= 0.5).mean(),
                         "p_min": pp[m].min(), "p_max": pp[m].max()}])
    pps.to_csv(OUT / "flat_pprice_quality.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"\n1. Model A (ตัวที่ใช้ต่อ: A-dir = {best_dir}, A-price = {best_price})")
    print(pd.concat([a_dir, a_price])[lambda t: t["group"] == "ALL"].round(4).to_string(index=False))
    print(f"\n2. ทิศทาง 3 คำตอบ (p จาก E9-{best_e9})")
    print(three_tab[three_tab["group"] == "ALL"].round(4).to_string(index=False))
    print("\n3. ราคากฎค่ามัธยฐาน")
    print(price_tab.round(4).to_string(index=False))
    print("\np_price:", pps.round(4).to_dict("records")[0])
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
