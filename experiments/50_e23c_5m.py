"""E23c: โมเดล E23 บนภาพ 5m — งานสำรวจ เกณฑ์ใน results/e23c_preregistration.md · ไม่ใช้ตัดสิน

    python experiments/50_e23c_5m.py
→ outputs/experiments/E23c/ (มีราคาจริง ไม่อยู่ใน git) · results/e23c/ (ตัวเลขสรุปเท่านั้น)
"""
import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_rules
from forecast.features.images import build_images, windows

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
LENGTH, BLOCK, MIN_TRAIN = 80, 5, 20
BATCHES = (64, 4)
OUT, RES = OUTPUT_DIR / "experiments" / "E23c", ROOT / "results" / "e23c"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def images_5m(g, bars_by_symbol, rules, length=LENGTH):
    """แถวที่มีแท่งครบ length ก่อนจุดตัด + ภาพ 56×56 ของแถวนั้น · ใช้เฉพาะแท่งที่จบไม่เกินจุดตัด"""
    import torch
    from torch.nn import functional as F
    keep, imgs = [], []
    for sym, h in g.groupby("symbol"):
        bars = bars_by_symbol[sym]
        first = bars["date"].min()
        h = h[h["date"] >= first]
        ok = np.array([len(w) == length for w in windows(bars, h, rules, length)], dtype=bool)
        h = h[ok]
        keep.append(h)
        imgs.append(build_images(bars, h, rules, "scalogram", length, normalize=False))
    out, x = pd.concat(keep), np.concatenate(imgs)
    order = np.lexsort((out["symbol"].to_numpy(), out["date"].to_numpy()))
    return out.iloc[order].reset_index(drop=True), F.interpolate(torch.from_numpy(x[order]), size=(56, 56), mode="area").numpy()


def period(dates):
    d = pd.to_datetime(pd.Series(dates).astype(str).str[:10])
    return np.select([d <= "2026-10-12", d <= "2026-10-29"], ["walk-forward", "holdout"], "ข้อมูลใหม่")


def score(x, names, n_boot, boot_rmse):
    rows = []
    for name in names:
        h = x[x[f"pt_{name}"].notna()]
        for sess, th in SESSIONS:
            s = h[h["session"] == sess]
            sets = [(th, "ทั้งช่วง", s), (f"KBANK {th}", "ทั้งช่วง", s[s["symbol"] == "KBANK"]), (f"ADVANC {th}", "ทั้งช่วง", s[s["symbol"] == "ADVANC"])]
            sets += [(th, p, s[s["period"] == p]) for p in ("walk-forward", "holdout", "ข้อมูลใหม่")]
            for lab, span, g in sets:
                if g.empty:
                    continue
                pe, dates = g["p_end"].to_numpy(float), g["date"].astype(str).to_numpy()
                e = pe - g[f"pt_{name}"].to_numpy(float)
                r = {"model": name, "group": lab, "span": span, "n": len(g), "n_days": g["date"].nunique(), "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()), "mean_abs_shift": float((g[f"pt_{name}"] - g["p_ref"]).abs().mean())}
                for base, tag in [("P(c)", "pc"), ("PR-A", "pra")]:
                    if name != base:
                        r.update(zip([f"d_rmse_{tag}", f"lo_{tag}", f"hi_{tag}"], boot_rmse(e, pe - g[f"pt_{base}"].to_numpy(float), dates, 0.95, n_boot)))
                rows.append(r)
    return pd.DataFrame(rows)


def pr_a_points(ds_wf, exps, rules, s47, s18, tz):
    a = pd.read_csv(OUTPUT_DIR / "experiments" / "E23" / "points.csv")[KEYS + ["pt_PR-A"]]
    c = pd.read_csv(OUTPUT_DIR / "experiments" / "E22" / "new_rows.csv")[KEYS + ["pt_PR-A"]]
    for t in (a, c):
        t["date"] = pd.to_datetime(t["date"], utc=True).dt.tz_convert(tz)
    b = s47.report_window(ds_wf, 10, exps, rules, {"morning": 0.0, "afternoon": 0.0}, s18)[KEYS + ["pt_PR-A"]]
    out = pd.concat([a, b, c], ignore_index=True)
    assert not out[KEYS].duplicated().any()
    return out


def main():
    from forecast.data.io import load_bars
    from forecast.evaluation.splits import walk_forward_splits
    from forecast.models.cnn1d import train_device
    s48, s47, s18 = [_module(n) for n in ["48_e23_skip_fx.py", "47_e22_shrink_k.py", "18_score_round4.py"]]
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    ds_wf, exps, s31, s33, s38, make_fx = s48.prepare(rules)
    new_ds = s48._load_new(rules)
    d, _ = s31.aligned(new_ds, exps)
    d, _ = s48.add_fx(d, make_fx(new_ds), s38)
    d = d[d["direction"].notna() & d["p_end"].notna() & d["p_ref"].notna()]
    cols, fx_cols = s48.columns(s33, exps, s38)
    bars = {s: load_bars(s, rules, None, "5m") for s in d["symbol"].unique()}
    dev = train_device()
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    base, tests = {}, []
    for s, _ in SESSIONS:
        g, im = images_5m(d[d["session"] == s], bars, rules)
        blocks = [te for _, te in walk_forward_splits(g["date"].to_numpy(), BLOCK, MIN_TRAIN, 0)]
        base[s] = (g, im, blocks)
        print(s, "แถว", len(g), "วัน", g["date"].nunique(), "บล็อกทดสอบ", len(blocks), flush=True)
    preds = []
    for batch in BATCHES:
        s48.BATCH = batch
        for s, _ in SESSIONS:
            g, im, blocks = base[s]
            preds.append(s48.run_session(g, im, cols[s], fx_cols, s, blocks, dev, s48.SEEDS, s48.EPOCHS, {}).assign(batch=batch))
    preds = pd.concat(preds, ignore_index=True)
    rows = pd.concat([base[s][0][KEYS + ["p_ref", "p_end"]] for s, _ in SESSIONS], ignore_index=True)
    x = rows.merge(preds[preds["batch"] == BATCHES[0]][KEYS], on=KEYS).merge(pr_a_points(ds_wf, exps, rules, s47, s18, tz), on=KEYS, how="left")
    assert x["pt_PR-A"].notna().all(), "ไม่มี PR-A บางแถว"
    x["pt_P(c)"] = x["p_ref"]
    x["period"] = period(x["date"])
    names = ["P(c)", "PR-A"]
    for batch in BATCHES:
        h = x[KEYS + ["p_ref"]].merge(preds[preds["batch"] == batch].drop(columns="batch"), on=KEYS)
        assert (h[KEYS].to_numpy() == x[KEYS].to_numpy()).all()
        for n in s48.NAMES:
            x[f"pt_{n}_b{batch}"] = h["p_ref"].to_numpy(float) * np.exp(h[f"dz_{n}"].to_numpy(float))
            names.append(f"{n}_b{batch}")
    x.to_csv(OUT / "rows.csv", index=False)
    t = score(x, names, n_boot, s38.boot_rmse)
    t.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(t[t["span"] == "ทั้งช่วง"].drop(columns="span").round(3).to_string(index=False))
    print(t[(t["span"] != "ทั้งช่วง")][["model", "group", "span", "n", "rmse", "mae"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
