"""ตรวจ overfit ทั้งโครงการ — เกณฑ์: results/overfit_plan.md · รายงานเท่านั้น ไม่เปลี่ยนโมเดล

    python experiments/42_overfit_audit.py [--part train pbo curves time holdout]
→ results/overfit/*.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import ast
import importlib
import warnings
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation import runner
from forecast.evaluation.splits import walk_forward_splits
from forecast.models import get_model
from forecast.models.e9 import target_rows
from forecast.models.nested_tune import MODELS, build, proba

warnings.filterwarnings("ignore")
s17, s18, s30, s31, s33 = [importlib.import_module(n) for n in ["17_label_variants", "18_score_round4", "30_e16_up_notup", "31_e14_finetune", "33_e15_models"]]
EXP, RES = OUTPUT_DIR / "experiments", ROOT / "results" / "overfit"
KEYS = ["date", "symbol", "session"]
SESSIONS = ("morning", "afternoon")


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_pred(path, tz):
    x = pd.read_csv(path)
    x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_convert(tz)
    return x


def labeled(g, target):
    """→ แถวของ g ที่ใช้กับเป้าหมาย + y"""
    m, y = target_rows(g, target)
    return g[m].assign(_y=y), m


def e0_wf(d, target, rules):
    """สัดส่วนจากวันก่อนบล็อกทดสอบ แยกหุ้นและรอบ → แถว KEYS + e0"""
    ev = rules["evaluation"]
    out = []
    for s in SESSIONS:
        g, _ = labeled(d[d["session"] == s], target)
        all_days = d.loc[d["session"] == s, "date"].to_numpy()
        for _, te in walk_forward_splits(all_days, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]):
            start = np.min(te)
            for sym, h in g.groupby("symbol"):
                prior, cur = h[h["date"] < start], h[h["date"].isin(te)]
                if len(prior) and len(cur):
                    out.append(cur[KEYS + ["_y"]].assign(e0=prior["_y"].mean()))
    return pd.concat(out, ignore_index=True)


def train_e0_ll(g_tr):
    """log loss ของสัดส่วนในข้อมูลฝึก บนข้อมูลฝึกนั้น แยกหุ้น"""
    rate = g_tr.groupby("symbol")["_y"].transform("mean").to_numpy()
    return ll(g_tr["_y"].to_numpy(float), rate).mean()


def test_gain(pred, e0, s, conf=.95):
    j = e0[e0["session"] == s].merge(pred[pred["session"] == s][KEYS + ["p"]], on=KEYS)
    y = j["_y"].to_numpy(float)
    gain = ll(y, j["e0"]) - ll(y, j["p"])
    m, lo, hi = s33.boot_ci(gain, j["date"].astype(str).to_numpy(), conf, 2000)
    return {"test_gain": m, "test_lo": lo, "test_hi": hi, "n_test": len(j), "test_ll": ll(y, j["p"]).mean(), "test_e0_ll": ll(y, j["e0"]).mean()}


# ---------- ส่วน 1: train เทียบ test ----------
def starts_by_session():
    t = pd.read_csv(EXP / "E15" / "tuning.csv")
    return {s: sorted(t.loc[t["session"] == s, "start"].unique()) for s in SESSIONS}


def lr_job(g, v, target, s, start, rules):
    """fit สเปกเดิม ณ start แล้ววัดบนข้อมูลฝึก → log loss ฝึก"""
    group_by = v.get("group_by", ["symbol", "session"])
    X = runner.design_matrix(g, runner.select_features(g, v), group_by)
    day = g["date"].dt.strftime("%Y-%m-%d").to_numpy()
    tr = (day < start) & g["up"].notna().to_numpy()
    kw = dict(v.get("params", {}))
    if v["model"] == "e12_lr":
        kw["permutation"] = False
    m = get_model(v["model"], rules["quantiles"], **kw).fit(X[tr], g["up"].to_numpy()[tr], g["z"].to_numpy()[tr], g[tr])
    sel = tr & (g["session"] == s).to_numpy()
    p = m.predict(X[sel], g[sel])["p_up"].to_numpy()
    h = g[sel].assign(_p=p)
    h, _ = labeled(h, target)
    y = h["_y"].to_numpy(float)
    return {"session": s, "start": start, "train_ll": ll(y, h["_p"]).mean(), "train_e0_ll": train_e0_ll(h), "n_train": len(h)}


def lr_family(ds, rules, tz, starts):
    exps = load_experiments()
    fd = next(v for v in exps["E12"]["variants"] if v["name"] == "fd_lr")
    un, _, _ = s30.variants()
    fl = {x["name"]: x for x in exps["E9FLAT"]["variants"]}
    specs = [("A-ud", fd, "C1235/E12_fd_lr_predictions.csv", "updown"),
             ("A-un", un, "E16/a_un_predictions.csv", "up_notup"),
             ("A-price (PR-A)", fl["aprice_lr"], "C1235/E9FLAT_aprice_lr_predictions.csv", "flat_price"),
             ("p_price (PR-A)", fl["pprice_lr"], "C1235/E9FLAT_pprice_lr_predictions.csv", "price_updown")]
    rows = []
    for name, v, file, target in specs:
        group_by = v.get("group_by", ["symbol", "session"])
        tasks = []
        for _, g in (ds.groupby(group_by) if group_by else [((), ds)]):
            for s in g["session"].unique():
                tasks += [(g.reset_index(drop=True), s, st) for st in starts[s]]
        res = pd.DataFrame(Parallel(n_jobs=-1)(delayed(lr_job)(g, v, target, s, st, rules) for g, s, st in tasks))
        res = res.groupby(["session", "start"], as_index=False).apply(lambda t: pd.Series({"train_ll": np.average(t["train_ll"], weights=t["n_train"]), "train_e0_ll": np.average(t["train_e0_ll"], weights=t["n_train"])}), include_groups=False)
        pred = load_pred(EXP / file, tz).rename(columns={"p_up": "p"})
        e0 = e0_wf(ds, target, rules)
        for s in SESSIONS:
            r = res[res["session"] == s]
            rows.append({"family": "LR ที่ใช้/ตัวเทียบ", "model": name, "target": target, "session": s, "n_ckpt": len(r), "train_gain": (r["train_e0_ll"] - r["train_ll"]).mean(), "inner_gain": np.nan, **test_gain(pred, e0, s)})
        print(name, "เสร็จ", flush=True)
    return rows


def nested_job(g, name, row, target):
    feats = ast.literal_eval(row["features"]) if row["features"].startswith("[") else []
    params = ast.literal_eval(row["params"])
    h, _ = labeled(g, target)
    day = h["date"].dt.strftime("%Y-%m-%d").to_numpy()
    tr = h[day < row["start"]]
    y = tr["_y"].to_numpy()
    m = build(name, params, 0).fit(tr[feats].to_numpy(float), y)
    p = proba(m, tr[feats].to_numpy(float))
    return {"session": row["session"], "period": row["period"], "model": name, "train_ll": ll(y.astype(float), p).mean(), "train_e0_ll": train_e0_ll(tr),
            "inner_gain": row["e0_ll"] - row["inner_ll"]}


def nested_family(d, rules, tz, family, tuning, predictions, target):
    t = pd.read_csv(tuning)
    t["params"], t["features"] = t["params"].astype(str), t["features"].astype(str)
    t = t[t["inner_ll"].notna()]
    jobs = [(d[d["session"] == r["session"]].reset_index(drop=True), r["model"], r) for _, r in t.iterrows()]
    res = pd.DataFrame(Parallel(n_jobs=-1)(delayed(nested_job)(g, n, r, target) for g, n, r in jobs))
    all_p = load_pred(predictions, tz)
    all_p = all_p[all_p["split"] == "wf"]
    e0 = e0_wf(d, target, rules)
    rows = []
    for s in SESSIONS:
        for name in MODELS:
            r = res[(res["session"] == s) & (res["model"] == name)]
            rows.append({"family": family, "model": name, "target": target, "session": s, "n_ckpt": len(r), "train_gain": (r["train_e0_ll"] - r["train_ll"]).mean(), "inner_gain": r["inner_gain"].mean(),
                         **test_gain(all_p[all_p["model"] == name], e0, s)})
    return rows


def verdict(r):
    if r["train_gain"] >= .010 and r["test_gain"] <= 0:
        return "overfit ชัด"
    if r["train_gain"] - r["test_gain"] >= .005:
        return "มีช่องว่าง"
    return "ไม่พบช่องว่างมาก"


def part_train(ds, d, rules, tz):
    starts = starts_by_session()
    rows = lr_family(ds, rules, tz, starts)
    rows += nested_family(d, rules, tz, "E15 ทิศทาง", EXP / "E15" / "tuning.csv", EXP / "E15" / "predictions.csv", "up_notup")
    print("E15 เสร็จ", flush=True)
    for target in ("flat_price", "price_updown"):
        rows += nested_family(d, rules, tz, f"E17b {target}", EXP / "E17b" / f"{target}_tuning.csv", EXP / "E17b" / f"{target}_predictions.csv", target)
        print("E17b", target, "เสร็จ", flush=True)
    t = pd.DataFrame(rows)
    t["gap"] = t["train_gain"] - t["test_gain"]
    t["verdict"] = t.apply(verdict, axis=1)
    t.to_csv(RES / "train_vs_test.csv", index=False, encoding="utf-8-sig")
    print(t[["family", "model", "session", "n_ckpt", "train_gain", "inner_gain", "test_gain", "test_lo", "test_hi", "gap", "verdict"]].round(4).to_string(index=False), flush=True)


def inner_vs_test(d, rules, tz):
    """E14 และ E17a: ใช้ inner_ll ที่บันทึกไว้ เทียบ log loss ทดสอบแถวเดียวกัน"""
    rows = []
    e0 = e0_wf(d, "up_notup", rules)
    for name, file in [("E14 D", "d_predictions.csv"), ("E14 D+C", "dc_predictions.csv")]:
        p = load_pred(EXP / "E14" / file, tz).rename(columns={"p_up": "p"})
        for s in SESSIONS:
            r = test_gain(p, e0, s)
            rows.append({"model": name, "session": s, "inner_ll": p.loc[p["session"] == s, "inner_ll"].mean(), "test_ll": r["test_ll"], "test_gain": r["test_gain"], "test_lo": r["test_lo"], "test_hi": r["test_hi"]})
    ft = load_pred(EXP / "E17a" / "ft_predictions.csv", tz).rename(columns={"p": "p"})
    for target in ("flat_price", "price_updown"):
        e = e0_wf(d, target, rules)
        for kind in ft["kind"].unique():
            p = ft[(ft["target"] == target) & (ft["kind"] == kind)]
            for s in SESSIONS:
                r = test_gain(p, e, s)
                rows.append({"model": f"E17a {kind} {target}", "session": s, "inner_ll": p.loc[p["session"] == s, "inner_ll"].mean(), "test_ll": r["test_ll"], "test_gain": r["test_gain"], "test_lo": r["test_lo"], "test_hi": r["test_hi"]})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "inner_vs_test.csv", index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False), flush=True)


# ---------- ส่วน 2: PBO ----------
def own_ll(path, tz, target="updown"):
    """log loss รายแถวจากไฟล์ผล โดยใช้ป้ายในไฟล์นั้น"""
    x = load_pred(path, tz)
    d = x["direction"].to_numpy(float)
    if target == "updown":
        m = np.isfinite(d) & (d != 0)
    else:
        m = np.isfinite(d)
    x = x[m]
    return x[KEYS].assign(ll=ll((x["direction"] == 1).to_numpy(float), x["p_up"]))


def long_from_dir(dirname, tz, skip=("bootstrap", "summary")):
    out = {}
    for f in sorted((EXP / dirname).glob("*_predictions.csv")):
        try:
            out[f"{dirname}/{f.stem.replace('_predictions', '')}"] = own_ll(f, tz)
        except KeyError:
            pass
    return out


def pbo_row(name, models, s=10):
    rows = []
    for sess in SESSIONS:
        per_day = {}
        for m, x in models.items():
            h = x[x["session"] == sess]
            if len(h):
                per_day[m] = h.assign(day=h["date"].dt.strftime("%Y-%m-%d")).groupby("day")["ll"].mean()
        if len(per_day) < 3:
            continue
        t = pd.DataFrame(per_day).dropna()
        if len(t) < 2 * s:
            rows.append({"group": name, "session": sess, "n_models": len(per_day), "n_days": len(t), "pbo": np.nan, "spread": np.nan, "best": "", "worst": ""})
            continue
        mean = t.mean()
        rows.append({"group": name, "session": sess, "n_models": t.shape[1], "n_days": len(t), "pbo": s33.pbo(-t, s=s), "spread": mean.max() - mean.min(), "best": mean.idxmin(), "worst": mean.idxmax()})
    return rows


def part_pbo(ds, d, rules, tz):
    fam = {}
    for n in ["E9", "E12", "E10", "EZOO", "E1", "E2", "E6", "E7", "E7s1"]:
        fam[n] = long_from_dir(n, tz)
    rows = []
    for n in ["E9", "E12", "E10", "EZOO"]:
        rows += pbo_row(n, fam[n])
    rows += pbo_row("รอบ 1–3 รวม", {k: v for f in fam.values() for k, v in f.items()})

    wide = load_pred(EXP / "E15" / "predictions_all.csv", tz)
    wide = wide[wide["split"] == "wf"]
    e15 = {}
    un = e0_wf(d, "up_notup", rules)
    for name, h in wide.groupby("model"):
        j = un.merge(h[KEYS + ["p"]], on=KEYS)
        e15[f"E15/{name}"] = j[KEYS].assign(ll=ll(j["_y"].to_numpy(float), j["p"]))
    rows += pbo_row("E15", e15)

    r7 = dict(e15)
    r7["E0-un"] = un[KEYS].assign(ll=ll(un["_y"].to_numpy(float), un["e0"]))
    for name, file, col in [("A-un", "E16/a_un_predictions.csv", "p_up"), ("A-dec", "E16/a_dec_predictions.csv", "p_up"), ("A-ud", "C1235/E12_fd_lr_predictions.csv", "p_up"),
                            ("E14 D", "E14/d_predictions.csv", "p_up"), ("E14 D+C", "E14/dc_predictions.csv", "p_up"), ("E18 A-un+fx", "E18/un_predictions.csv", "p_up")]:
        h = load_pred(EXP / file, tz)
        j = un.merge(h[KEYS + [col]], on=KEYS)
        r7[name] = j[KEYS].assign(ll=ll(j["_y"].to_numpy(float), j[col]))
    c = load_pred(EXP / "E14" / "cprime_predictions.csv", tz)
    for col in [x for x in c.columns if x.startswith("p_")]:
        j = un.merge(c[KEYS + [col]], on=KEYS)
        r7[f"E14 {col}"] = j[KEYS].assign(ll=ll(j["_y"].to_numpy(float), j[col]))
    rows += pbo_row("ทิศทางรอบ 7 รวม", r7)

    rows += pbo_row("E17b", price_family(d, rules, tz, with_a=False))
    rows += pbo_row("ราคารอบ 7 รวม", price_family(d, rules, tz, with_a=True))

    e19 = {}
    for tag, name in [("1m", "E19_explore"), ("5m", "E19_explore_5m")]:
        x = load_pred(EXP / name / "predictions.csv", tz)
        y = (x["direction"] == 1).to_numpy(float)
        x = x.assign(ll=ll(y, x["p_up"]), session="afternoon")
        for (a, k), h in x.groupby(["arch", "kind"]):
            e19[f"{tag}/{a}/{k}"] = h[KEYS + ["ll"]]
        e19["E0-un"] = x.groupby(["date", "symbol"], as_index=False).first().assign(ll=lambda t: ll((t["direction"] == 1).to_numpy(float), t["p_e0"]))[KEYS + ["ll"]]
    rows += pbo_row("E19 (6 วัน ใช้สรุปไม่ได้)", e19, s=6)

    t = pd.DataFrame(rows)
    t.to_csv(RES / "pbo.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False), flush=True)


def price_family(d, rules, tz, with_a):
    """log loss รวม A-price + p_price รายแถว (แถวละครึ่งหนึ่งของคู่เป้าหมาย รวมเป็นรายวันด้วย mean ของแต่ละเป้าหมาย)"""
    eq, up = e0_wf(d, "flat_price", rules), e0_wf(d, "price_updown", rules)
    out = {}

    def combine(name, pe, pu):
        a = eq.merge(pe, on=KEYS)
        b = up.merge(pu, on=KEYS)
        out[name] = pd.concat([a[KEYS].assign(ll=ll(a["_y"].to_numpy(float), a["p"])), b[KEYS].assign(ll=ll(b["_y"].to_numpy(float), b["p"]))], ignore_index=True)

    fe, fu = [load_pred(EXP / "E17b" / f"{t}_predictions.csv", tz) for t in ("flat_price", "price_updown")]
    fe, fu = fe[fe["split"] == "wf"], fu[fu["split"] == "wf"]
    for name in fe["model"].unique():
        combine(f"E17b/{name}", fe.loc[fe["model"] == name, KEYS + ["p"]], fu.loc[fu["model"] == name, KEYS + ["p"]])
    if with_a:
        pe, pu = [load_pred(EXP / "C1235" / f"E9FLAT_{n}_predictions.csv", tz).rename(columns={"p_up": "p"})[KEYS + ["p"]] for n in ("aprice_lr", "pprice_lr")]
        combine("PR-A", pe, pu)
        ft = load_pred(EXP / "E17a" / "ft_predictions.csv", tz)
        for kind in ft["kind"].unique():
            combine(f"E17a {kind}", ft.loc[(ft["kind"] == kind) & (ft["target"] == "flat_price"), KEYS + ["p"]], ft.loc[(ft["kind"] == kind) & (ft["target"] == "price_updown"), KEYS + ["p"]])
    return out


# ---------- ส่วน 3: learning curve ----------
def curve_row(path):
    c = pd.read_csv(path).dropna(subset=["val_loss"])
    if c.empty:
        return None
    rows = []
    fold_col = [x for x in ("fold_train_end", "fold") if x in c.columns][0]
    for _, h in c.groupby([fold_col, "seed"]):
        h = h.sort_values("epoch")
        i = h["val_loss"].to_numpy().argmin()
        after = h["train_loss"].to_numpy()[i:]
        rows.append({"best": h["epoch"].iloc[i], "last": h["epoch"].iloc[-1], "train_drop_after": after[0] - after[-1], "val_rise_after": h["val_loss"].iloc[-1] - h["val_loss"].iloc[i]})
    r = pd.DataFrame(rows)
    return {"experiment": path.parent.name, "model": path.stem.replace("_curves", ""), "runs": len(r), "best_epoch_median": r["best"].median(), "last_epoch_median": r["last"].median(),
            "share_best_le3": (r["best"] <= 3).mean(), "train_drop_after": r["train_drop_after"].median(), "val_rise_after": r["val_rise_after"].median()}


def part_curves():
    rows = [curve_row(p) for p in sorted(EXP.glob("*/*curves*.csv")) if p.parent.name != "E19_smoke"]
    t = pd.DataFrame([r for r in rows if r])
    t["overfit_early"] = (t["best_epoch_median"] <= 3) & (t["train_drop_after"] > 0)
    t.to_csv(RES / "cnn_curves.csv", index=False, encoding="utf-8-sig")
    print(t.round(3).to_string(index=False), flush=True)


# ---------- ส่วน 4: ความนิ่งตามเวลา ----------
def part_time(d, rules, tz):
    un, ud = e0_wf(d, "up_notup", rules), e0_wf(d, "updown", rules)
    e15 = load_pred(EXP / "E15" / "predictions.csv", tz)
    specs = [("A-ud", "morning", load_pred(EXP / "C1235/E12_fd_lr_predictions.csv", tz).rename(columns={"p_up": "p"}), ud),
             ("A-ud", "afternoon", load_pred(EXP / "C1235/E12_fd_lr_predictions.csv", tz).rename(columns={"p_up": "p"}), ud),
             ("A-un", "morning", load_pred(EXP / "E16/a_un_predictions.csv", tz).rename(columns={"p_up": "p"}), un),
             ("A-un", "afternoon", load_pred(EXP / "E16/a_un_predictions.csv", tz).rename(columns={"p_up": "p"}), un),
             ("E15 MLP", "afternoon", e15[(e15["model"] == "mlp") & (e15["split"] == "wf")], un)]
    rows = []
    for name, s, p, e0 in specs:
        j = e0[e0["session"] == s].merge(p[p["session"] == s][KEYS + ["p"]], on=KEYS)
        days = np.sort(j["date"].unique())
        for i, part in enumerate(np.array_split(days, 3)):
            h = j[j["date"].isin(part)]
            y = h["_y"].to_numpy(float)
            m, lo, hi = s33.boot_ci(ll(y, h["e0"]) - ll(y, h["p"]), h["date"].astype(str).to_numpy(), .95, 2000)
            rows.append({"model": name, "session": s, "part": i + 1, "from": str(part[0])[:10], "to": str(part[-1])[:10], "n": len(h), "gain": m, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "time_stability.csv", index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False), flush=True)


# ---------- ส่วน 5: holdout ----------
def part_holdout(d, rules, tz):
    h = pd.read_csv(ROOT / "results" / "e16" / "holdout.csv")
    h = h[h["group"].isin(["เช้า", "บ่าย"])].set_index(["model", "group"])
    th = {"morning": "เช้า", "afternoon": "บ่าย"}
    e0 = h.xs("E0-un", level="model")["logloss_un"]
    un = e0_wf(d, "up_notup", rules)
    rows = []
    e15h = pd.read_csv(ROOT / "results" / "e15" / "holdout.csv")
    e15 = load_pred(EXP / "E15" / "predictions.csv", tz)
    for name, s, file in [("A-ud", "morning", "C1235/E12_fd_lr_predictions.csv"), ("A-ud", "afternoon", "C1235/E12_fd_lr_predictions.csv"), ("A-un", "morning", "E16/a_un_predictions.csv"), ("A-un", "afternoon", "E16/a_un_predictions.csv")]:
        r = test_gain(load_pred(EXP / file, tz).rename(columns={"p_up": "p"}), un, s)
        rows.append({"model": name, "session": s, "wf_gain_un": r["test_gain"], "holdout_gain_un": e0[th[s]] - h.loc[(name, th[s]), "logloss_un"], "holdout_n": h.loc[(name, th[s]), "n"]})
    r = test_gain(e15[(e15["model"] == "mlp") & (e15["split"] == "wf")], un, "afternoon")
    m = e15h[(e15h["model"] == "mlp") & (e15h["session"] == "afternoon")].iloc[0]
    rows.append({"model": "E15 MLP", "session": "afternoon", "wf_gain_un": r["test_gain"], "holdout_gain_un": e0["บ่าย"] - m["ll"], "holdout_n": m["n"]})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "holdout_gain.csv", index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="*", default=["train", "inner", "pbo", "curves", "time", "holdout"])
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = runner.load_or_build_dataset(rules, as_of=end, tag="_wf")
    d, _ = s31.aligned(ds, load_experiments())
    print(f"dataset {len(ds)} แถว ถึง {ds['date'].max().date()}", flush=True)
    if "curves" in args.part:
        part_curves()
    if "time" in args.part:
        part_time(d, rules, tz)
    if "holdout" in args.part:
        part_holdout(d, rules, tz)
    if "inner" in args.part:
        inner_vs_test(d, rules, tz)
    if "pbo" in args.part:
        part_pbo(ds, d, rules, tz)
    if "train" in args.part:
        part_train(ds, d, rules, tz)


if __name__ == "__main__":
    main()
