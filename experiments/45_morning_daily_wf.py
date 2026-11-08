"""A-un / A-ud / E0-un รอบเช้า แบบฝึกใหม่รายวัน (รายงานเท่านั้น ไม่ใช้เปลี่ยนโมเดล)

    python experiments/45_morning_daily_wf.py
→ results/e16_morning_daily/ (rows.csv, summary.csv, pairs.csv, up_share.csv, repro_check.csv, report.md)

สเปกเดียวกับ E16 · dataset tag _new (อ่านอย่างเดียว) · แต่ละวันทดสอบฝึกด้วยข้อมูลก่อนวันนั้นเท่านั้น
ไม่อ่านโมเดลที่ล็อก ไม่แก้ rules.yaml · วันนิ่งนับเป็นไม่ขึ้น
"""
import importlib.util
import pickle

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import PROCESSED_DIR, ROOT, load_rules, raw_source_dir
from forecast.data.io import load_bars
from forecast.evaluation.runner import design_matrix
from forecast.features.dataset import select_features
from forecast.labels.make_labels import labels_for
from forecast.models import get_model

spec = importlib.util.spec_from_file_location("m30", ROOT / "experiments" / "30_e16_up_notup.py")
s30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s30)

KEYS = ["date", "symbol", "session"]
RES = ROOT / "results" / "e16_morning_daily"
NEW_DAYS, HOLDOUT_DAYS = 6, 10
SESSION = "morning"
MODELS = ["A-un", "A-ud", "E0-un"]
BACKTEST_UP = 0.431      # results/e16_report.md: วันขึ้นจริงรอบเช้าใน walk-forward


def morning_rows(ds):
    d = ds[ds["session"] == SESSION].copy()
    assert d.duplicated(["date", "symbol"]).sum() == 0
    return d.sort_values(["symbol", "date"])


def predict(d, variant, rules, test_days, train_before):
    """train_before(วันทดสอบ) → วันแรกที่ห้ามใช้ฝึก · สเปก E12 fd_lr: group_by [session] = ฝึกรวมสองหุ้นในรอบเช้า"""
    group_by = variant.get("group_by", ["symbol", "session"])
    assert group_by == ["session"], group_by
    g = d.reset_index(drop=True)
    X = design_matrix(g, select_features(g, variant), group_by)
    labeled = g["up"].notna()
    out = []
    for day in test_days:
        te = g["date"] == day
        tr = (g["date"] < train_before(day)) & labeled
        assert te.sum() == 2 and g.loc[tr, "date"].max() < day
        model = get_model(variant["model"], rules["quantiles"], **variant.get("params", {}))
        model.fit(X[tr], g["up"].to_numpy()[tr], g["z"].to_numpy()[tr], g[tr])
        p = model.predict(X[te], g[te])
        out.append(g.loc[te, KEYS].join(p[["p_up"]]).assign(trained_until=g.loc[tr, "date"].max()))
    return pd.concat(out, ignore_index=True)


def e0_un(d, test_days, train_before):
    out = []
    for sym, g in d.groupby("symbol"):
        for day in test_days:
            if not (g["date"] == day).any():
                continue
            h = g[(g["date"] < train_before(day)) & g["direction"].notna()]
            out.append({"date": day, "symbol": sym, "session": SESSION, "p_up": float((h["direction"] == 1).mean())})
    return pd.DataFrame(out)


def run_all(d, rules, un, fd, test_days, train_before):
    parts = {"A-un": predict(d, {**un, "params": {**un["params"], "permutation": False}}, rules, test_days, train_before),
             "A-ud": predict(d, {**fd, "params": {**fd.get("params", {}), "permutation": False}}, rules, test_days, train_before),
             "E0-un": e0_un(d, test_days, train_before)}
    x = d[d["date"].isin(test_days)][KEYS + ["p_start", "p_end", "direction"]].copy()
    for name, p in parts.items():
        cols = KEYS + ["p_up"] + (["trained_until"] if name == "A-un" else [])
        x = x.merge(p[cols].rename(columns={"p_up": f"p_{name}"}), on=KEYS, validate="1:1")
    assert x["direction"].notna().all()
    return x.sort_values(["date", "symbol"]).reset_index(drop=True)


def score(x):
    x = x.copy()
    y = (x["direction"] == 1).to_numpy(float)
    x["actual_up"] = y.astype(int)
    for m in MODELS:
        x[f"pred_{m}"] = (x[f"p_{m}"] > 0.5).astype(int)
        x[f"ok_{m}"] = (x[f"pred_{m}"] == x["actual_up"]).astype(int)
        x[f"ll_{m}"] = s30.losses(x[f"p_{m}"].to_numpy(), y)[0]
    return x


def settrade_labels(rules):
    raw = raw_source_dir(rules, "settrade")
    rows = []
    for sym in rules["symbols"]:
        lab = labels_for(load_bars(sym, rules, raw, "1m"), rules, "1m")
        rows.append(lab[lab["session"] == SESSION][["date", "symbol", "p_start", "p_end", "direction"]])
    return pd.concat(rows).rename(columns={"p_start": "p_start_1m", "p_end": "p_end_1m", "direction": "direction_1m"})


def summarize(x, periods, n_boot):
    rows, pair_rows = [], []
    for label, mask in periods.items():
        g = x[mask(x)]
        for m in MODELS:
            ci = s30.s18.boot_mean(g[f"ok_{m}"], g["date"], n_boot)
            rows.append({"period": label, "model": m, "n": len(g), "n_days": g["date"].nunique(),
                         "n_correct": int(g[f"ok_{m}"].sum()), "acc": g[f"ok_{m}"].mean(), "acc_lo": ci[1], "acc_hi": ci[2],
                         "logloss": g[f"ll_{m}"].mean(), "pred_up_rate": g[f"pred_{m}"].mean()})
        for a, b in [("A-un", "A-ud"), ("A-un", "E0-un"), ("A-ud", "E0-un")]:
            da = s30.s18.boot_mean(g[f"ok_{a}"] - g[f"ok_{b}"], g["date"], n_boot)
            dl = s30.s18.boot_mean(g[f"ll_{a}"] - g[f"ll_{b}"], g["date"], n_boot)
            pair_rows.append({"period": label, "pair": f"{a} − {b}", "d_acc": da[0], "d_acc_lo": da[1], "d_acc_hi": da[2],
                              "d_ll": dl[0], "d_ll_lo": dl[1], "d_ll_hi": dl[2]})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def up_share(x, periods):
    rows = []
    for label, mask in periods.items():
        g = x[mask(x)]
        for sym, h in [("รวม", g)] + list(g.groupby("symbol")):
            n_up, n = int(h["actual_up"].sum()), len(h)
            rows.append({"period": label, "symbol": sym, "n": n, "n_up": n_up, "n_flat": int((h["direction"] == 0).sum()),
                         "up_share": n_up / n, "backtest": BACKTEST_UP, "diff": n_up / n - BACKTEST_UP})
    return pd.DataFrame(rows)


def repro(d, rules, un, fd, new_days, hold_days):
    """ตรวจว่า pipeline นี้ซ้ำผล E16 ได้: ฝึกครั้งเดียวก่อนช่วงทดสอบ เทียบ results/e16/{new_data,holdout}.csv"""
    rows = []
    for name, days, f in [("new_data", new_days, "new_data.csv"), ("holdout", hold_days, "holdout.csv")]:
        first = days[0]
        x = score(run_all(d, rules, un, fd, days, lambda _: first))
        ref = pd.read_csv(ROOT / "results" / "e16" / f)
        ref = ref[ref["group"] == "เช้า"].set_index("model")
        for m in MODELS:
            rows.append({"set": name, "model": m, "acc": x[f"ok_{m}"].mean(), "acc_e16": ref.loc[m, "acc_up_notup"],
                         "logloss": x[f"ll_{m}"].mean(), "logloss_e16": ref.loc[m, "logloss_un"]})
    return pd.DataFrame(rows)


def md_table(df):
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)) + "\n"


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    with open(PROCESSED_DIR / "dataset_yfinance_60m_pcoff_atc_c1235_new.pkl", "rb") as f:
        ds = pickle.load(f)
    d = morning_rows(ds.reset_index(drop=True))
    days = sorted(d["date"].unique())
    new_days, hold_days = days[-NEW_DAYS:], days[-(NEW_DAYS + HOLDOUT_DAYS):-NEW_DAYS]
    assert str(new_days[0])[:10] == "2026-10-30" and str(new_days[-1])[:10] == "2026-11-06", new_days
    assert str(hold_days[0])[:10] == "2026-10-14" and str(hold_days[-1])[:10] == "2026-10-29", hold_days
    un, fd, _ = s30.variants()
    n_boot = rules["evaluation"]["n_boot"]

    check = repro(d, rules, un, fd, new_days, hold_days)
    check.to_csv(RES / "repro_check.csv", index=False, encoding="utf-8-sig")
    print(check.round(4).to_string(index=False), flush=True)

    test_days = hold_days + new_days
    x = score(run_all(d, rules, un, fd, test_days, lambda day: day))
    x["period"] = np.where(x["date"].isin(new_days), "30 ต.ค.–6 พ.ย.", "14–29 ต.ค. (holdout)")

    m1 = settrade_labels(rules)
    x = x.merge(m1, on=["date", "symbol"], how="left")
    known = x["direction_1m"].notna()
    x["label_mismatch_1m"] = known & ((x["direction"] == 1) != (x["direction_1m"] == 1))
    print(f"เทียบ label กับ Settrade 1m ได้ {int(known.sum())} แถว · ขึ้น/ไม่ขึ้น ไม่ตรง {int(x['label_mismatch_1m'].sum())} แถว", flush=True)

    keep = ["date", "symbol", "period", "p_start", "p_end", "direction", "actual_up", "direction_1m", "label_mismatch_1m", "trained_until"]
    for m in MODELS:
        keep += [f"p_{m}", f"pred_{m}", f"ok_{m}", f"ll_{m}"]
    out = x[keep].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["trained_until"] = pd.to_datetime(out["trained_until"]).dt.strftime("%Y-%m-%d")
    out.to_csv(RES / "rows.csv", index=False, encoding="utf-8-sig")

    periods = {"30 ต.ค.–6 พ.ย.": lambda t: t["date"].isin(new_days),
               "14–29 ต.ค. (holdout)": lambda t: t["date"].isin(hold_days),
               "รวม 14 ต.ค.–6 พ.ย.": lambda t: t["date"].isin(test_days)}
    summ, pr = summarize(x, periods, n_boot)
    summ.to_csv(RES / "summary.csv", index=False, encoding="utf-8-sig")
    pr.to_csv(RES / "pairs.csv", index=False, encoding="utf-8-sig")
    shares = up_share(x, periods)
    shares.to_csv(RES / "up_share.csv", index=False, encoding="utf-8-sig")

    word = {1: "ขึ้น", 0: "ไม่ขึ้น"}
    real = {1.0: "ขึ้น", 0.0: "นิ่ง", -1.0: "ลง"}
    t1 = pd.DataFrame({"วันที่": out["date"], "หุ้น": out["symbol"], "จริง": out["direction"].map(real)})
    for m in MODELS:
        t1[f"{m} P(ขึ้น)"] = out[f"p_{m}"].map("{:.3f}".format)
        t1[f"{m} ทาย"] = out[f"pred_{m}"].map(word) + out[f"ok_{m}"].map({1: " ✓", 0: " ✗"})
    new_rows = out["period"] == "30 ต.ค.–6 พ.ย."
    t2 = summ.assign(acc=lambda s: (s["acc"] * 100).map("{:.1f}%".format), logloss=summ["logloss"].map("{:.4f}".format),
                     CI=lambda s: [f"[{a:.0%}, {b:.0%}]" for a, b in zip(s["acc_lo"], s["acc_hi"])],
                     ถูก=lambda s: s["n_correct"].astype(str) + "/" + s["n"].astype(str))[["period", "model", "ถูก", "acc", "CI", "logloss"]]
    t3 = shares.assign(up_share=shares["up_share"].map("{:.1%}".format), backtest=shares["backtest"].map("{:.1%}".format),
                       diff=(shares["diff"] * 100).map("{:+.1f} จุด".format))[["period", "symbol", "n_up", "n", "n_flat", "up_share", "backtest", "diff"]]
    (RES / "report.md").write_text(
        "# A-un / A-ud / E0-un รอบเช้า ฝึกใหม่รายวัน (รายงานเท่านั้น)\n\n"
        "`experiments/44_morning_daily_wf.py` · สเปก E16 · dataset `_new` · แต่ละวันฝึกด้วยข้อมูลก่อนวันนั้น · วันนิ่งนับเป็นไม่ขึ้น\n\n"
        "## 1. รายแถว 30 ต.ค.–6 พ.ย.\n\n" + md_table(t1[new_rows.to_numpy()]) +
        "\n## 1ข. รายแถว holdout 14–29 ต.ค.\n\n" + md_table(t1[(~new_rows).to_numpy()]) +
        "\n## 2. สรุป\n\n" + md_table(t2) +
        "\n## 3. สัดส่วนวันขึ้นจริง\n\n" + md_table(t3), encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(t1[new_rows.to_numpy()].to_string(index=False))
    print(t2.to_string(index=False))
    print(pr.round(4).to_string(index=False))
    print(t3.to_string(index=False))
    print(out.loc[out["label_mismatch_1m"], ["date", "symbol", "p_start", "p_end", "direction", "direction_1m"]].to_string(index=False))
    print("trained_until ล่าสุด:", out["trained_until"].max(), "· วันทดสอบล่าสุด:", out["date"].max())


if __name__ == "__main__":
    main()
