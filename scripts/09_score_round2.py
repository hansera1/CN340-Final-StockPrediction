"""ตัวชี้วัดรอบที่ 2 จาก predictions ที่มีอยู่ (ไม่ฝึกใหม่) — NEXT_STEPS ข้อ 1, 2 (ราคา) และตาราง E10

    python scripts/09_score_round2.py
    → outputs/reports/round2/*.csv

ตัวชี้วัดหลัก: ความแม่นทิศทางแบบไม่นับ flat / flat = ผิด, log loss ขึ้น vs ลง (เฉพาะวันไม่ flat)
และ 95% CI ของผลต่างเทียบ E0 ด้วย block bootstrap รายวัน แยกหุ้น/รอบ และรวม
E0 มีสองแบบ: E0 รอบแรก (P(ขึ้น) = สัดส่วนขึ้นของทุกวัน → ทาย "ลง" เกือบทุกวัน) และ E0-ud (สัดส่วนขึ้นในวันไม่ flat)
โมเดลรอบแรกฝึก "ขึ้น vs ไม่ขึ้น" — คิดคะแนนด้วย P(ขึ้น) เดิมตามจริง ไม่ปรับ
"""
import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules
from forecast.evaluation.bootstrap import block_bootstrap_diff
from forecast.evaluation.metrics import row_losses, tick_size
from forecast.evaluation.runner import load_or_build_dataset

EXP = OUTPUT_DIR / "experiments"
OUT = OUTPUT_DIR / "reports" / "round2"
KEYS = ["date", "symbol", "session"]

ROUND1 = [("E0 baseline (รอบแรก)", "E0", "baseline"), ("E1 independent", "E1", "independent"),
          ("E2 cascade", "E2", "cascade"), ("E6 pooled", "E6", "pooled"),
          ("E7a scalogram ขั้น 0", "E7", "e7a_scalogram_s0"), ("E7b แท่งเทียน ขั้น 0", "E7", "e7b_candle_s0"),
          ("E7a scalogram ขั้น 1", "E7s1", "e7a_scalogram_s1"), ("E7b แท่งเทียน ขั้น 1", "E7s1", "e7b_candle_s1")]
ROUND2 = [("E0-ud baseline (ขึ้น vs ลง)", "E9", "baseline_ud"), ("E9-LR", "E9", "lr"), ("E9-LRx", "E9", "lrx"),
          ("E9-GBM", "E9", "gbm"), ("E9-ENS", "E9", "ens")]
ARCH_ORDER = ["resnet18", "densenet121", "inception_v3", "vgg16_bn", "alexnet"]
GROUPS = [None, ("KBANK", "morning"), ("KBANK", "afternoon"), ("ADVANC", "morning"), ("ADVANC", "afternoon")]


def load(exp, variant, tz):
    path = EXP / exp / f"{variant}_predictions.csv"
    if not path.exists():
        return None
    p = pd.read_csv(path)
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def e10_models():
    out = []
    for arch in ARCH_ORDER:
        for st in ["s0", "s1", "s2", "s3", "s2_x20", "s3_x20"]:
            if (EXP / "E10" / f"{arch}_{st}_predictions.csv").exists():
                out.append((f"E10 {arch} {st.upper()}", "E10", f"{arch}_{st}"))
    return out


def gname(g):
    return "ALL" if g is None else f"{g[0]}/{g[1]}"


def subset(p, g):
    return p if g is None else p[(p["symbol"] == g[0]) & (p["session"] == g[1])]


def scores(p, q):
    l = row_losses(p, q)
    d = p.loc[l.index]
    return {
        "n": len(d), "n_nonflat": int(l["err_nf"].notna().sum()),
        "acc_nf": 1 - l["err_nf"].mean(), "acc_fw": 1 - l["err_fw"].mean(),
        "logloss_ud": l["logloss_ud"].mean(), "logloss_old": l["logloss"].mean(),
        "pred_up_share": float((d["p_up"] >= 0.5).mean()),
        "mae_baht": l["abs_err_baht"].mean(), "mae_ticks": l["abs_err_ticks"].mean(),
        "coverage50": l["cov50"].mean(), "coverage80": l["cov80"].mean(),
        "width50_baht": l["width50_baht"].mean(), "width80_baht": l["width80_baht"].mean(),
        "pinball": l["pinball"].mean(),
    }


def boot(a, b, q, ev, g):
    rows = {}
    for metric in ("err_nf", "err_fw", "logloss_ud"):
        r = block_bootstrap_diff(subset(a, g), subset(b, g), q, metric, ev["block_days"], ev["n_boot"])
        if r is None:
            continue
        sign = -1 if metric.startswith("err") else 1      # err → รายงานเป็นผลต่างความแม่น (บวก = ดีกว่า)
        lo, hi = sorted((sign * r["ci_low"], sign * r["ci_high"]))
        key = {"err_nf": "acc_nf", "err_fw": "acc_fw", "logloss_ud": "logloss_ud"}[metric]
        rows.update({f"d_{key}": sign * r["mean_diff"], f"d_{key}_lo": lo, f"d_{key}_hi": hi})
    return rows


def seed_table(models, preds, q):
    rows = []
    for label, _, _ in models:
        p = preds[label]
        cols = [c for c in p if c.startswith("p_up_seed")]
        if not cols:
            continue
        per = [scores(p.assign(p_up=p[c]), q) for c in cols]
        acc, ll = [s["acc_nf"] for s in per], [s["logloss_ud"] for s in per]
        avg = scores(p, q)
        rows.append({"model": label, "n_seeds": len(cols), "acc_nf_mean": np.mean(acc), "acc_nf_min": min(acc),
                     "acc_nf_max": max(acc), "logloss_ud_mean": np.mean(ll), "logloss_ud_min": min(ll),
                     "logloss_ud_max": max(ll), "acc_nf_avgprob": avg["acc_nf"], "logloss_ud_avgprob": avg["logloss_ud"]})
    return pd.DataFrame(rows)


def quarterly(ds):
    d = ds[ds["direction"].notna()].copy()
    d["quarter"] = d["date"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    t = d.groupby(["symbol", "session", "quarter"])["direction"].agg(
        n="size", up=lambda s: (s == 1).mean(), flat=lambda s: (s == 0).mean(), down=lambda s: (s == -1).mean())
    t["up_minus_down"] = t["up"] - t["down"]
    nf = d[d["direction"] != 0]
    t["up_share_nonflat"] = nf.groupby(["symbol", "session", "quarter"])["direction"].apply(lambda s: (s == 1).mean())
    return t.reset_index()


def price_detail(p, ds, q):
    """E9 ราคา: แยก |ผิดพลาด| ของราคาจุด (= P(c)) เป็นส่วน gap (P(เริ่ม) − P(c)) และส่วนในช่วง (P(ปลาย) − P(เริ่ม))
    + ทางเลือกเสริม: เลื่อนราคาจุด 1 ช่องตามทิศที่ทายเมื่อ P > 0.55 (หรือ < 0.45)"""
    m = p.merge(ds[KEYS + ["p_start"]], on=KEYS, how="left")
    m = m[m["p_end"].notna()].copy()
    tick = tick_size(m["p_ref"])
    m["shift"] = np.select([m["p_up"] > 0.55, m["p_up"] < 0.45], [tick, -tick], 0.0)
    rows = []
    for g in GROUPS:
        s = subset(m, g)
        t = tick_size(s["p_ref"])
        sh = s["shift"].to_numpy()
        rows.append({
            "group": gname(g), "n": len(s),
            "mae_point_baht": (s["p_end"] - s["p_ref"]).abs().mean(),
            "mae_point_ticks": ((s["p_end"] - s["p_ref"]).abs() / t).mean(),
            "mean_abs_gap_part_baht": (s["p_start"] - s["p_ref"]).abs().mean(),
            "mean_abs_window_part_baht": (s["p_end"] - s["p_start"]).abs().mean(),
            "mae_shift1tick_baht": (s["p_end"] - (s["p_ref"] + sh)).abs().mean(),
            "share_shifted": float((np.asarray(sh) != 0).mean()) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    rules = load_rules()
    tz, q, ev = rules["timezone"], rules["quantiles"], rules["evaluation"]
    OUT.mkdir(parents=True, exist_ok=True)
    models = [m for m in ROUND1 + ROUND2 + e10_models() if (EXP / m[1] / f"{m[2]}_predictions.csv").exists()]
    preds = {label: load(exp, v, tz) for label, exp, v in models}
    e0, e0ud = preds["E0 baseline (รอบแรก)"], preds.get("E0-ud baseline (ขึ้น vs ลง)")

    # วันทดสอบร่วมของทุกโมเดล (E10 refit ทุก 20 วันแต่วันทดสอบชุดเดียวกัน)
    common = None
    for p in preds.values():
        k = set(map(tuple, p[KEYS].astype(str).to_numpy()))
        common = k if common is None else common & k
    for label in preds:
        p = preds[label]
        preds[label] = p[[tuple(r) in common for r in p[KEYS].astype(str).to_numpy()]].reset_index(drop=True)
    e0, e0ud = preds["E0 baseline (รอบแรก)"], preds.get("E0-ud baseline (ขึ้น vs ลง)")
    print(f"แถวทดสอบร่วม: {len(e0)} รอบ ({e0['date'].min().date()} – {e0['date'].max().date()})")

    rows = []
    for label, exp, v in models:
        for g in GROUPS:
            r = {"model": label, "exp": exp, "variant": v, "group": gname(g), **scores(subset(preds[label], g), q)}
            if label != "E0 baseline (รอบแรก)":
                r.update({f"vsE0_{k}": x for k, x in boot(preds[label], e0, q, ev, g).items()})
            if e0ud is not None and not label.startswith("E0"):
                r.update({f"vsE0ud_{k}": x for k, x in boot(preds[label], e0ud, q, ev, g).items()})
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics_round2.csv", index=False)

    # เทียบ E10 กับ E9 ตัวที่ดีที่สุด (ความแม่นไม่นับ flat รวม) และ E7a ขั้น 1
    e9 = table[(table["exp"] == "E9") & (table["group"] == "ALL") & (table["variant"] != "baseline_ud")]
    if len(e9):
        best9 = e9.sort_values("acc_nf", ascending=False)["model"].iat[0]
        refs = {"best_E9": preds[best9], "E7a_s1": preds.get("E7a scalogram ขั้น 1")}
        rows = []
        for label, exp, v in models:
            if exp != "E10":
                continue
            for g in GROUPS:
                r = {"model": label, "group": gname(g), "best_E9": best9}
                for name, ref in refs.items():
                    if ref is not None:
                        r.update({f"vs{name}_{k}": x for k, x in boot(preds[label], ref, q, ev, g).items()})
                rows.append(r)
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "e10_vs_e9_e7.csv", index=False)

    seed_table(models, preds, q).to_csv(OUT / "seeds.csv", index=False)

    ds = load_or_build_dataset(rules)
    quarterly(ds).to_csv(OUT / "direction_by_quarter.csv", index=False)

    if "E9-LR" in preds:
        price_detail(preds["E9-LR"], ds, q).to_csv(OUT / "e9_price_detail.csv", index=False)
    gbm = preds.get("E9-GBM")
    if gbm is not None:
        sh = gbm.filter(like="shap_").abs().groupby(gbm["session"]).mean().T
        sh.index = sh.index.str.replace("shap_", "")
        sh.round(5).to_csv(OUT / "e9_gbm_shap.csv")
        hp = pd.concat([preds["E9-LR"].groupby("session")["C"].value_counts().rename("n").reset_index()
                        .assign(param="C").rename(columns={"C": "value"}),
                        gbm.groupby("session")["min_data_in_leaf"].value_counts().rename("n").reset_index()
                        .assign(param="min_data_in_leaf").rename(columns={"min_data_in_leaf": "value"})])
        hp.to_csv(OUT / "e9_chosen_hparams.csv", index=False)

    show = ["model", "group", "n_nonflat", "acc_nf", "acc_fw", "logloss_ud", "mae_baht", "coverage80",
            "vsE0ud_d_acc_nf", "vsE0ud_d_acc_nf_lo", "vsE0ud_d_acc_nf_hi"]
    pd.set_option("display.width", 250)
    print(table[table["group"] == "ALL"][[c for c in show if c in table]].round(4).to_string(index=False))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
