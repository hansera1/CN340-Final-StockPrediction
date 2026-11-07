"""E23: CNN+ANN + skip ราคาอ้างอิง + feature ต่างประเทศรอบเช้า — เกณฑ์ใน results/e23_preregistration.md · รายงานเท่านั้น

    python experiments/48_e23_skip_fx.py --smoke     # ตรวจ pipeline 2 บล็อก ไม่คำนวณคะแนน · ห้ามอ้างเป็นผล
    python experiments/48_e23_skip_fx.py --part run
    python experiments/48_e23_skip_fx.py --part report
→ outputs/experiments/E23/ (มีราคาจริง ไม่อยู่ใน git) · results/e23/ (ตัวเลขสรุปเท่านั้น)
"""
import argparse
import importlib.util
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
KINDS = {"morning": ["skip_tab", "skip_tab_fx", "skip_cnn", "skip_cnn_fx"], "afternoon": ["skip_tab", "skip_cnn"]}
REPRESENTATIVE = {"morning": "skip_cnn_fx", "afternoon": "skip_cnn"}
BLOCK, SEEDS, EPOCHS, PATIENCE, BATCH, LR, WD, VAL = 20, (0, 1, 2), 30, 3, 64, 1e-3, 1e-2, 0.2
RES, OUT = ROOT / "results" / "e23", OUTPUT_DIR / "experiments" / "E23"
NAMES = ["skip_tab", "skip_tab_fx", "skip_cnn", "skip_cnn_fx"]
PAIRS = [("skip_cnn", "skip_tab", "CNN ช่วยไหม"), ("skip_tab_fx", "skip_tab", "ต่างประเทศช่วยไหม (ไม่มี CNN)"), ("skip_cnn_fx", "skip_cnn", "ต่างประเทศช่วยไหม (มี CNN)")]


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_net(use_cnn, n_tab):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 8), nn.ReLU()) if use_cnn else None
            self.tab = nn.Sequential(nn.Linear(n_tab, 8), nn.ReLU(), nn.Dropout(0.3))
            self.head = nn.Linear(8 + 1 + (8 if use_cnn else 0), 1)
            nn.init.zeros_(self.head.weight)     # เริ่มจากส่วนที่ขยับ = 0 → ราคาทาย = P(c)
            nn.init.zeros_(self.head.bias)

        def forward(self, img, sym, tab):
            parts = [self.tab(tab), sym]
            if self.cnn is not None:
                parts.append(self.cnn(img))
            return self.head(torch.cat(parts, 1)).squeeze(-1)

    return Net()


def fit_predict(use_cnn, img, sym, tab, z, tr, te, dates, dev, seeds, epochs, fold, curves):
    """ฝึกจากแถว tr เท่านั้น → ส่วนที่ขยับ (หน่วย ln) ของแถว te"""
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from torch.nn import functional as F
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(tab[tr])
    sc = StandardScaler().fit(imp.transform(tab[tr]))
    t_all = torch.as_tensor(np.clip(sc.transform(imp.transform(tab)), -10, 10), dtype=torch.float32, device=dev)
    s_all = torch.as_tensor(sym[:, None], dtype=torch.float32, device=dev)
    i_all = None
    if use_cnn:
        mean, std = img[tr].mean(axis=(0, 2, 3), keepdims=True), img[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-8
        i_all = torch.as_tensor((img - mean) / std, dtype=torch.float32, device=dev)
    scale = float(np.std(z[tr])) or 1.0
    y = torch.as_tensor(np.nan_to_num(z) / scale, dtype=torch.float32, device=dev)
    ud = np.unique(dates[tr])
    cut = ud[int(len(ud) * (1 - VAL))]
    a, b = torch.as_tensor(np.flatnonzero(tr & (dates < cut))), torch.as_tensor(np.flatnonzero(tr & (dates >= cut)), device=dev)
    it = torch.as_tensor(np.flatnonzero(te), device=dev)
    pick = lambda idx: (i_all[idx] if i_all is not None else None, s_all[idx], t_all[idx])
    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = make_net(use_cnn, tab.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
        g = torch.Generator().manual_seed(seed)
        best, state, bad = np.inf, None, 0
        for ep in range(epochs):
            net.train()
            tl, tn = 0.0, 0
            for idx in a[torch.randperm(len(a), generator=g)].split(BATCH):
                idx = idx.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(net(*pick(idx)), y[idx])
                loss.backward()
                opt.step()
                tl, tn = tl + loss.item() * len(idx), tn + len(idx)
            net.eval()
            with torch.no_grad():
                v = F.mse_loss(net(*pick(b)), y[b]).item()
            curves.append({"fold_train_end": fold, "seed": seed, "epoch": ep + 1, "train_loss": tl / tn, "val_loss": v})
            if v < best - 1e-5:
                best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            out.append(net(*pick(it)).cpu().numpy() * scale)
    return np.mean(out, axis=0)


def add_fx(d, fx, s38):
    """คอลัมน์ต่างประเทศของแถวเช้า · ค่าว่าง = 0"""
    FX = s38.FX
    out = d.merge(s38.add_asia_gap(fx)[["date"] + FX], on="date", how="left")
    missing = out.loc[out["session"] == "morning", FX].isna().sum().to_dict()
    out[FX] = out[FX].fillna(0.0)
    return out, missing


def run_session(g, im, base_cols, fx_cols, session, blocks, dev, seeds, epochs, curves):
    """g: แถวของรอบเดียว → แถว KEYS + ส่วนที่ขยับต่อแบบ"""
    dates, sym = g["date"].to_numpy(), g["d_sym_advanc"].to_numpy(float)
    z = np.log(g["p_end"].to_numpy(float) / g["p_ref"].to_numpy(float))
    base, with_fx = g[base_cols].to_numpy(float), g[base_cols + fx_cols].to_numpy(float)
    rows = []
    for n, te_days in enumerate(blocks):
        t0 = time.perf_counter()
        start = np.min(te_days)
        tr, te = (dates < start) & np.isfinite(z), np.isin(dates, te_days)
        fold = f"{session}_{str(start)[:10]}"
        r = g.loc[te, KEYS].copy()
        for kind in KINDS[session]:
            use_cnn, use_fx = "cnn" in kind, kind.endswith("_fx")
            r[f"dz_{kind}"] = fit_predict(use_cnn, im, sym, with_fx if use_fx else base, z, tr, te, dates, dev, seeds, epochs, fold, curves.setdefault(kind, []))
        rows.append(r)
        print(session, fold, f"{n + 1}/{len(blocks)}", f"{time.perf_counter() - t0:.0f} วิ", flush=True)
    return pd.concat(rows, ignore_index=True)


def small_images(ds_reset, d, rules, tag):
    import torch
    from forecast.evaluation.runner import images_for
    from torch.nn import functional as F
    img = images_for(ds_reset, rules, "scalogram", tag=tag)
    pos = ds_reset[KEYS].assign(_i=np.arange(len(ds_reset))).merge(d[KEYS].assign(_j=np.arange(len(d))), on=KEYS).sort_values("_j")["_i"].to_numpy()
    assert len(pos) == len(d)
    return F.interpolate(torch.from_numpy(img[pos]).float().div_(255), size=(56, 56), mode="area").numpy()


def session_rows(d, im, session):
    m = np.flatnonzero((d["session"] == session).to_numpy())
    return d.iloc[m].reset_index(drop=True), im[m]


def prepare(rules):
    from forecast.evaluation.runner import load_or_build_dataset
    from forecast.features.overnight import build_features
    s17, s31, s33, s37, s38 = [_module(n) for n in ["17_label_variants.py", "31_e14_finetune.py", "33_e15_models.py", "37_e18_overnight.py", "38_e18b_gap_shift.py"]]
    assert rules["label"]["p_ref_morning"] == "official_close" and rules["label"]["price_1630"] == "closing_auction" and rules["sessions"]["afternoon"]["cutoff"] == "12:35"
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(rules["timezone"]) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    exps = load_experiments()
    tables, hourly = s37.load_tables()
    make_fx = lambda data: build_features(pd.DatetimeIndex(np.sort(data["date"].unique())), tables, hourly)
    return ds, exps, s31, s33, s38, make_fx


def columns(s33, exps, s38):
    feats = s33.feature_sets(exps)
    return {s: [c for c in feats[s] if c != "d_sym_advanc"] for s, _ in SESSIONS}, list(s38.FX)


def predict_window(data, tag, days, rules, exps, s31, s33, s38, make_fx, dev):
    """ฝึกครั้งเดียวจากวันก่อนช่วง days วันท้าย → ส่วนที่ขยับของช่วงนั้น · รายงานเท่านั้น"""
    d, _ = s31.aligned(data, exps)
    d, _ = add_fx(d, make_fx(data), s38)
    cols, fx_cols = columns(s33, exps, s38)
    im = small_images(data.reset_index(drop=True), d, rules, tag)
    out = []
    for s, _ in SESSIONS:
        g, gi = session_rows(d, im, s)
        out.append(run_session(g, gi, cols[s], fx_cols, s, [np.sort(g["date"].unique())[-days:]], dev, SEEDS, EPOCHS, {}))
    return pd.concat(out, ignore_index=True)


def run(rules, smoke=False):
    from forecast.evaluation.splits import walk_forward_splits
    from forecast.models.cnn1d import train_device
    ds, exps, s31, s33, s38, make_fx = prepare(rules)
    d, _ = s31.aligned(ds, exps)
    d, missing = add_fx(d, make_fx(ds), s38)
    print("ค่าว่างต่างประเทศที่แทน 0:", missing, flush=True)
    cols, fx_cols = columns(s33, exps, s38)
    im = small_images(ds.reset_index(drop=True), d, rules, "_e20wf")
    dev = train_device()
    ev = rules["evaluation"]
    out_dir = OUT.with_name("E23_smoke") if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds, epochs = ((0,), 2) if smoke else (SEEDS, EPOCHS)
    curves, parts = {}, []
    for s, _ in SESSIONS:
        g, gi = session_rows(d, im, s)
        blocks = [te for _, te in walk_forward_splits(g["date"].to_numpy(), BLOCK, ev["min_train_days"], ev["holdout_days"])]
        parts.append(run_session(g, gi, cols[s], fx_cols, s, blocks[:2] if smoke else blocks, dev, seeds, epochs, curves))
    pd.concat(parts, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    for k, c in curves.items():
        pd.DataFrame(c).to_csv(out_dir / f"{k}_curves.csv", index=False)
    if smoke:
        print("** SMOKE: ตรวจ pipeline เท่านั้น ห้ามอ้างเป็นผล **")
        return
    new_ds = _load_new(rules)
    predict_window(ds, "_e20wf", 10, rules, exps, s31, s33, s38, make_fx, dev).to_csv(OUT / "holdout_dz.csv", index=False)
    predict_window(new_ds, "_e23new", 6, rules, exps, s31, s33, s38, make_fx, dev).to_csv(OUT / "new_dz.csv", index=False)
    print("ครบ", flush=True)


def _load_new(rules):
    from forecast.evaluation.runner import load_or_build_dataset
    new_ds = load_or_build_dataset(rules, tag="_new")
    nd = np.sort(new_ds["date"].unique())
    assert str(nd[-6])[:10] == "2026-10-30" and str(nd[-1])[:10] == "2026-11-06", nd[-7:]
    return new_ds


def add_points(x, dz):
    """ราคาทาย = P(c) × exp(ส่วนที่ขยับ) · แบบที่ไม่มีในรอบนั้นเป็น NaN"""
    x = x.merge(dz, on=KEYS, how="inner")
    for n in NAMES:
        x[f"pt_{n}"] = x["p_ref"].to_numpy(float) * np.exp(x[f"dz_{n}"].to_numpy(float))
    return x


def score_available(x, names, n_boot, s47, s38, s31):
    """คะแนนต่อแบบต่อรอบ เฉพาะรอบที่แบบนั้นมีผล"""
    out = []
    for name in names:
        ok = x[f"pt_{name}"].notna().to_numpy()
        out.append(s47.score(x[ok].reset_index(drop=True), [name], n_boot, s38.boot_rmse, s31.boot_ci))
    return pd.concat(out, ignore_index=True)


def verdict(t):
    q, rows = t.set_index(["model", "group"]), []
    for sess, th in SESSIONS:
        m = REPRESENTATIVE[sess]
        a = q.loc[(m, th)]
        k1 = bool(a["d_rmse_pra_hi"] < 0)
        k2 = all(q.loc[(m, f"{s} {th}"), "d_rmse_pra_lo"] <= 0 for s in ("KBANK", "ADVANC"))
        k3 = bool(a["d_mae_pra_lo"] <= 0)
        rows.append({"session": th, "model": m, "c1_rmse": k1, "c2_stocks": k2, "c3_mae": k3, "pass": k1 and k2 and k3})
    return pd.DataFrame(rows)


def pair_table(x, n_boot, s38, s31):
    rows = []
    for a, b, q in PAIRS:
        for sess, th in SESSIONS:
            h = x[(x["session"] == sess) & x[f"pt_{a}"].notna() & x[f"pt_{b}"].notna()]
            if h.empty:
                continue
            ea, eb, dates = (h["p_end"] - h[f"pt_{a}"]).to_numpy(float), (h["p_end"] - h[f"pt_{b}"]).to_numpy(float), h["date"].to_numpy()
            r = {"question": q, "a": a, "b": b, "session": th, "n": len(h)}
            r.update(zip(["d_rmse", "lo", "hi"], s38.boot_rmse(ea, eb, dates, 0.997, n_boot)))
            r.update(zip(["lo95", "hi95"], s38.boot_rmse(ea, eb, dates, 0.95, n_boot)[1:]))
            rows.append(r)
    return pd.DataFrame(rows)


def report(rules):
    s42, s47, s18 = [_module(n) for n in ["42_overfit_audit.py", "47_e22_shrink_k.py", "18_score_round4.py"]]
    ds, exps, s31, s33, s38, make_fx = prepare(rules)
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    RES.mkdir(parents=True, exist_ok=True)
    d, _ = s31.aligned(ds, exps)
    eq, pp, zq = [s18.load("C1235", n, tz) for n in ["E9FLAT_aprice_lr", "E9FLAT_pprice_lr", "E9_baseline_ud"]]
    x = d[KEYS + ["p_ref", "p_end"]].merge(eq[KEYS + ["p_up"]].rename(columns={"p_up": "eq_A"}), on=KEYS).merge(pp[KEYS + ["p_up"]].rename(columns={"p_up": "up_A"}), on=KEYS).merge(zq[KEYS], on=KEYS)
    x = x.merge(s47.cond_means(d), on=KEYS)
    x = x[x["p_end"].notna() & x["p_ref"].notna() & x["z_up"].notna() & x["z_dn"].notna()].sort_values(KEYS).reset_index(drop=True)
    x = s47.attach_points(x, s18)
    dz = pd.read_csv(OUT / "predictions.csv")
    dz["date"] = pd.to_datetime(dz["date"], utc=True).dt.tz_convert(tz)
    n0 = len(x)
    x = add_points(x, dz)
    assert len(x) == n0 == 2188, (len(x), n0)
    x.to_csv(OUT / "points.csv", index=False)
    base = s47.score(x, ["PR-A", "P(c)"], n_boot, s38.boot_rmse, s31.boot_ci)
    t = pd.concat([base, score_available(x, NAMES, n_boot, s47, s38, s31)], ignore_index=True)
    got = t[t["group"].isin(["เช้า", "บ่าย"]) & t["model"].isin(["PR-A", "P(c)"])].set_index(["model", "group"])["rmse"].round(3).to_dict()
    assert got[("PR-A", "เช้า")] == 3.083 and got[("PR-A", "บ่าย")] == 1.830 and got[("P(c)", "เช้า")] == 3.073 and got[("P(c)", "บ่าย")] == 1.817, got
    v, pairs = verdict(t), pair_table(x, n_boot, s38, s31)
    curve = pd.DataFrame([{k: r[k] for k in ("model", "runs", "best_epoch_median", "last_epoch_median", "share_best_le3", "train_drop_after", "val_rise_after")} for r in [s42.curve_row(OUT / f"{n}_curves.csv") for n in NAMES]])
    for name, tab in [("metrics", t), ("verdict", v), ("pairs", pairs), ("curves", curve)]:
        tab.to_csv(RES / f"{name}.csv", index=False, encoding="utf-8-sig")
    windows = {}
    for name, data, days, file in [("holdout", ds, 10, "holdout_dz.csv"), ("new_data", _load_new(rules), 6, "new_dz.csv")]:
        w = s47.report_window(data, days, exps, rules, {"morning": 0.0, "afternoon": 0.0}, s18)
        z = pd.read_csv(OUT / file)
        z["date"] = pd.to_datetime(z["date"], utc=True).dt.tz_convert(tz)
        w = add_points(w, z)
        if name == "new_data":
            w.to_csv(OUT / "new_rows.csv", index=False)
        windows[name] = pd.concat([s47.score(w, ["PR-A", "P(c)"], n_boot, s38.boot_rmse, s31.boot_ci), score_available(w, NAMES, n_boot, s47, s38, s31)], ignore_index=True)
        windows[name].to_csv(RES / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    show = ["model", "group", "n", "rmse", "mae", "d_rmse_pra", "d_rmse_pra_lo", "d_rmse_pra_hi", "d95_rmse_pra_lo", "d95_rmse_pra_hi", "d_mae_pra", "d_mae_pra_lo", "d_mae_pra_hi", "d_rmse_pc", "d_rmse_pc_lo", "d_rmse_pc_hi"]
    print(t[show].round(3).to_string(index=False))
    print(v.to_string(index=False))
    print(pairs.round(4).to_string(index=False))
    print(curve.round(4).to_string(index=False))
    for k, tab in windows.items():
        print("---", k)
        print(tab[tab["group"].isin(["เช้า", "บ่าย"])][["model", "group", "n", "rmse", "mae", "d_rmse_pra", "d_rmse_pra_lo", "d_rmse_pra_hi"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=[])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    rules = load_rules()
    if args.smoke:
        run(rules, smoke=True)
    if "run" in args.part:
        run(rules)
    if "report" in args.part:
        report(rules)


if __name__ == "__main__":
    main()
