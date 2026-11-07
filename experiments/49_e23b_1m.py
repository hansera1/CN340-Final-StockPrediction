"""E23b: โมเดล E23 บนภาพ 1m — งานสำรวจ เกณฑ์ใน results/e23b_preregistration.md · ไม่ใช้ตัดสิน

    python experiments/25_1m_scalogram_demo.py --all-days --session morning --window 80     # สร้างภาพก่อน
    python experiments/25_1m_scalogram_demo.py --all-days --session afternoon --window 80
    python experiments/49_e23b_1m.py
→ outputs/experiments/E23b/ (มีราคาจริง ไม่อยู่ใน git) · results/e23b/ (ตัวเลขสรุปเท่านั้น)
"""
import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_rules

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
MIN_TRAIN = 5
BATCHES = (64, 4)
DEMO = OUTPUT_DIR / "demo" / "scalogram_1m_dataset"
OUT, RES = OUTPUT_DIR / "experiments" / "E23b", ROOT / "results" / "e23b"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def join_images(d, meta, X):
    """แถว dataset ของรอบเดียว + ภาพ 1m ที่ตรงหุ้นและวัน · ราคาต้องตรงกับชุดภาพ"""
    rows = d.assign(_day=d["date"].astype(str).str[:10]).reset_index(drop=True)
    m = meta.assign(_i=np.arange(len(meta)))
    g = rows.merge(m[["symbol", "date", "_i", "price_end", "session_close"]].rename(columns={"date": "_day"}), on=["symbol", "_day"], how="inner", validate="one_to_one")
    assert len(g) == len(meta), (len(g), len(meta))
    assert np.allclose(g["p_ref"].astype(float), g["price_end"]) and np.allclose(g["p_end"].astype(float), g["session_close"]), "ราคาในชุดภาพไม่ตรงกับ dataset"
    g = g.sort_values(["date", "symbol"]).reset_index(drop=True)
    return g, X[g["_i"].to_numpy()]


def score(x, names):
    rows = []
    for name in names:
        h = x[x[f"pt_{name}"].notna()]
        for sess, th in SESSIONS:
            for lab, mask in [(th, h["session"] == sess), (f"KBANK {th}", (h["session"] == sess) & (h["symbol"] == "KBANK")), (f"ADVANC {th}", (h["session"] == sess) & (h["symbol"] == "ADVANC"))]:
                g = h[mask]
                if g.empty:
                    continue
                e = (g["p_end"] - g[f"pt_{name}"]).to_numpy(float)
                rows.append({"model": name, "group": lab, "n": len(g), "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()), "mean_abs_shift": float((g[f"pt_{name}"] - g["p_ref"]).abs().mean())})
    return pd.DataFrame(rows)


def main():
    import torch
    from torch.nn import functional as F
    from forecast.evaluation.splits import walk_forward_splits
    from forecast.models.cnn1d import train_device
    s48 = _module("48_e23_skip_fx.py")
    rules = load_rules()
    tz = rules["timezone"]
    _, exps, s31, s33, s38, make_fx = s48.prepare(rules)
    new_ds = s48._load_new(rules)
    d, _ = s31.aligned(new_ds, exps)
    d, _ = s48.add_fx(d, make_fx(new_ds), s38)
    cols, fx_cols = s48.columns(s33, exps, s38)
    dev = train_device()
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    base = {}
    for name, sess_rows in [(s, d[d["session"] == s]) for s, _ in SESSIONS]:
        folder = DEMO / name
        parts = [np.load(folder / f"{k}.npz") for k in ("train", "valid", "test")]
        X = np.concatenate([p["X"] for p in parts])
        meta = pd.concat([pd.read_csv(folder / f"{k}.csv") for k in ("train", "valid", "test")], ignore_index=True)
        assert meta["session"].eq(name).all()
        small = F.interpolate(torch.from_numpy(X).float(), size=(56, 56), mode="area").numpy()
        base[name] = join_images(sess_rows, meta, small)
    all_preds = []
    for batch in BATCHES:
        s48.BATCH = batch
        for name, _ in SESSIONS:
            g, im = base[name]
            blocks = [te for _, te in walk_forward_splits(g["date"].to_numpy(), 1, MIN_TRAIN, 0)]
            p = s48.run_session(g, im, cols[name], fx_cols, name, blocks, dev, s48.SEEDS, s48.EPOCHS, {})
            all_preds.append(p.assign(batch=batch))
    preds = pd.concat(all_preds, ignore_index=True)
    w = _module("47_e22_shrink_k.py")
    rows = pd.read_csv(OUTPUT_DIR / "experiments" / "E22" / "new_rows.csv")
    rows["date"] = pd.to_datetime(rows["date"], utc=True).dt.tz_convert(tz)
    tests = pd.concat([base[s][0] for s, _ in SESSIONS], ignore_index=True)[KEYS + ["p_ref", "p_end"]]
    x = tests.merge(preds[preds["batch"] == BATCHES[0]][KEYS], on=KEYS).merge(rows[KEYS + ["pt_PR-A"]], on=KEYS, how="left")
    assert x["pt_PR-A"].notna().all(), "ไม่มี PR-A บางแถว"
    x["pt_P(c)"] = x["p_ref"]
    for s, _ in SESSIONS:
        r = pd.read_csv(DEMO / s / "walk_forward_predictions.csv")[["symbol", "date", "pred_session_close"]].rename(columns={"date": "_day", "pred_session_close": "_res"})
        h = x.assign(_day=x["date"].astype(str).str[:10]).merge(r, on=["symbol", "_day"], how="left")
        x.loc[x["session"] == s, "pt_ResNet18_1m"] = h.loc[(h["session"] == s).to_numpy(), "_res"].to_numpy()
    names = ["P(c)", "PR-A", "ResNet18_1m"]
    for batch in BATCHES:
        z = preds[preds["batch"] == batch].drop(columns="batch")
        h = x[KEYS + ["p_ref"]].merge(z, on=KEYS)
        for n in s48.NAMES:
            col = f"{n}_b{batch}"
            x[f"pt_{col}"] = (h["p_ref"].to_numpy(float) * np.exp(h[f"dz_{n}"].to_numpy(float)))
            names.append(col)
    assert len(x) == 18, len(x)
    x.to_csv(OUT / "rows.csv", index=False)
    t = score(x, names)
    t.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
