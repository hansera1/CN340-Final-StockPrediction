"""E21: ablation baseline → ตาราง → 1D-CNN → scalogram → fusion — เกณฑ์ใน results/e21_preregistration.md

    python experiments/44_e21_ablation.py --smoke     # ตรวจ pipeline 2 บล็อก ไม่คำนวณคะแนน · ห้ามอ้างเป็นผล
    python experiments/44_e21_ablation.py --part run
    python experiments/44_e21_ablation.py --part report
→ outputs/experiments/E21/*.csv, results/e21/*.csv · ไม่ใช้ holdout · ไม่แตะระบบ live · รายงานเท่านั้น
"""
import argparse
import importlib
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.features.images import windows

warnings.filterwarnings("ignore")
OUT, RES = OUTPUT_DIR / "experiments" / "E21", ROOT / "results" / "e21"
KEYS = ["date", "symbol", "session"]
SESSIONS = ("morning", "afternoon")
LENGTH, BLOCK = 56, 20
SEEDS, EPOCHS, PATIENCE, BATCH, LR, WD, VAL = (0, 1, 2), 30, 3, 64, 1e-3, 1e-2, 0.2
TAB_C = 0.001
NETS = ["cnn1d_3", "cnn1d_ext", "cnn2d_small", "fusion"]
ORDER = ["e0", "return_lr", "tab_lr", "cnn1d_3", "cnn1d_ext", "cnn2d_small", "resnet18_s2", "fusion"]
PAIRS = [("return_lr", "e0"), ("tab_lr", "return_lr"), ("cnn1d_3", "tab_lr"), ("cnn1d_ext", "cnn1d_3"), ("cnn2d_small", "cnn1d_ext"),
         ("resnet18_s2", "cnn2d_small"), ("fusion", "tab_lr"), ("fusion", "cnn1d_ext")]
CLOSE_HOUR = 16.5


# ---------- ลำดับข้อมูล ----------
def add_channels(bars):
    """ช่องระดับแท่ง ใช้เฉพาะแท่งนั้นและแท่งก่อนหน้า · volume surprise เทียบเวลาเดียวกันของวันก่อนเท่านั้น"""
    b = bars.sort_values("bar_end").reset_index(drop=True).copy()
    c = b["close"].astype(float)
    o = b["open"].astype(float) if "open" in b else c
    h = b["high"].astype(float) if "high" in b else np.maximum(o, c)
    lo = b["low"].astype(float) if "low" in b else np.minimum(o, c)
    b["c_r"] = np.log(c).diff().fillna(0.0) * 100
    b["c_abs"] = b["c_r"].abs()
    b["c_lv"] = np.log1p(b["volume"].astype(float))
    b["c_oc"] = np.log(c / o) * 100
    b["c_hl"] = np.log(h / lo) * 100
    b["c_rv"] = b["c_r"].rolling(6, min_periods=2).std().fillna(0.0)
    end = pd.DatetimeIndex(b["bar_end"])
    slot = end.hour * 60 + end.minute
    past = b["c_lv"].groupby(slot).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    b["c_vs"] = (b["c_lv"] - past).fillna(0.0)
    b["c_ttc"] = (CLOSE_HOUR - (end.hour + end.minute / 60)) / 6.5
    return b


CHANNELS = ["c_r", "c_abs", "c_lv", "c_oc", "c_hl", "c_rv", "c_vs", "c_ttc"]


def build_seqs(bars, rows, rules, length=LENGTH):
    """→ (n, 8, length) ชิดขวา · ใช้เฉพาะแท่งที่ bar_end ≤ จุดตัดของแถว · log volume ปรับ z ภายในหน้าต่าง"""
    out = np.zeros((len(rows), len(CHANNELS), length), dtype=np.float32)
    for i, w in enumerate(windows(add_channels(bars), rows, rules, length)):
        if len(w) < 2:
            continue
        x = w[CHANNELS].to_numpy(float).T
        x[2] = (x[2] - x[2].mean()) / (x[2].std() + 1e-8)
        out[i, :, length - x.shape[1]:] = np.nan_to_num(x)
    return out


def return_features(seq):
    """ผลตอบแทน 1, 6, 28 แท่งล่าสุดจากช่อง r"""
    r = seq[:, 0]
    return np.stack([r[:, -1], r[:, -6:].sum(1), r[:, -28:].sum(1)], axis=1)


# ---------- โครงข่าย ----------
def make_net(kind, n_tab=0):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "cnn2d_small":
                self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                          nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            else:
                ch = 3 if kind == "cnn1d_3" else len(CHANNELS)
                self.body = nn.Sequential(nn.Conv1d(ch, 16, 5, padding=2), nn.ReLU(), nn.Dropout(0.3), nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.tab = nn.Sequential(nn.Linear(n_tab, 8), nn.ReLU(), nn.Dropout(0.3)) if n_tab else None
            self.head = nn.Linear(16 + 1 + (8 if n_tab else 0), 1)

        def forward(self, x, sym, tab):
            parts = [self.body(x), sym]
            if self.tab is not None:
                parts.append(self.tab(tab))
            return self.head(torch.cat(parts, 1)).squeeze(-1)

    return Net()


def train_one(kind, seed, x, sym, tab, y, a, b, dev, epochs, fold, curves):
    """ฝึกจากแถว a · early stopping บนแถว b → net ที่ดีที่สุด"""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed)
    net = make_net(kind, tab.shape[1] if tab is not None else 0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    ia, ib = torch.as_tensor(np.flatnonzero(a)), torch.as_tensor(np.flatnonzero(b)).to(dev)
    best, state, bad = np.inf, None, 0
    tb = tab if tab is not None else sym
    for ep in range(epochs):
        net.train()
        tl, tn = 0.0, 0
        for idx in ia[torch.randperm(len(ia), generator=g)].split(BATCH):
            idx = idx.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(net(x[idx], sym[idx], tb[idx]), y[idx])
            loss.backward()
            opt.step()
            tl, tn = tl + loss.item() * len(idx), tn + len(idx)
        net.eval()
        with torch.no_grad():
            v = F.binary_cross_entropy_with_logits(net(x[ib], sym[ib], tb[ib]), y[ib]).item()
        curves.append({"fold_train_end": fold, "seed": seed, "epoch": ep + 1, "train_loss": tl / tn, "val_loss": v})
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(state)
    return net.eval()


def fit_predict_net(kind, x_raw, sym_raw, tab_raw, y, tr, te, dates, dev, seeds, epochs, fold, curves):
    """normalize + imputer fit บนแถว tr เท่านั้น → P(ขึ้น) ของแถว te"""
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    axes = (0, 2, 3) if x_raw.ndim == 4 else (0, 2)
    mean, std = x_raw[tr].mean(axis=axes, keepdims=True), x_raw[tr].std(axis=axes, keepdims=True) + 1e-8
    x = torch.as_tensor((x_raw - mean) / std, dtype=torch.float32, device=dev)
    sym = torch.as_tensor(sym_raw[:, None], dtype=torch.float32, device=dev)
    tab = None
    if tab_raw is not None:
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(tab_raw[tr])
        sc = StandardScaler().fit(imp.transform(tab_raw[tr]))
        tab = torch.as_tensor(np.clip(sc.transform(imp.transform(tab_raw)), -10, 10), dtype=torch.float32, device=dev)
    yt = torch.as_tensor(np.where(y >= 0, y, 0), dtype=torch.float32, device=dev)
    ud = np.unique(dates[tr])
    cut = ud[int(len(ud) * (1 - VAL))]
    a, b = tr & (dates < cut), tr & (dates >= cut)
    ps = []
    it = torch.as_tensor(np.flatnonzero(te), device=dev)
    for s in seeds:
        net = train_one(kind, s, x, sym, tab, yt, a, b, dev, epochs, fold, curves)
        with torch.no_grad():
            ps.append(torch.sigmoid(net(x[it], sym[it], (tab if tab is not None else sym)[it])).cpu().numpy())
    return np.mean(ps, axis=0)


# ---------- รัน ----------
def prepare(rules):
    from forecast.evaluation.runner import load_or_build_dataset, load_bars
    s17, s31 = [importlib.import_module(n) for n in ("17_label_variants", "31_e14_finetune")]
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    exps = load_experiments()
    d, _ = s31.aligned(ds, exps)
    return ds, d, exps, load_bars


def run(rules, smoke=False):
    import torch
    from forecast.evaluation.runner import images_for
    from forecast.evaluation.splits import walk_forward_splits
    from forecast.models.cnn1d import train_device
    from forecast.models.e9 import target_rows
    from forecast.models.nested_tune import build, proba
    from torch.nn import functional as F
    s33 = importlib.import_module("33_e15_models")
    ds, d, exps, load_bars = prepare(rules)
    out_dir = OUT.with_name("E21_smoke") if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = s33.feature_sets(exps)
    seqs = np.zeros((len(d), len(CHANNELS), LENGTH), dtype=np.float32)
    for sym in d["symbol"].unique():
        m = np.flatnonzero((d["symbol"] == sym).to_numpy())
        seqs[m] = build_seqs(load_bars(sym, rules), d.iloc[m], rules)
    dd = ds.reset_index(drop=True)
    img = images_for(dd, rules, "scalogram", tag="_e20wf")
    pos = dd[KEYS].assign(_i=np.arange(len(dd))).merge(d[KEYS].assign(_j=np.arange(len(d))), on=KEYS).sort_values("_j")["_i"].to_numpy()
    assert len(pos) == len(d)
    small = F.interpolate(torch.from_numpy(img[pos]).float().div_(255), size=(56, 56), mode="area").numpy()
    del img
    dev = train_device()
    ev = rules["evaluation"]
    seeds, epochs = ((0,), 2) if smoke else (SEEDS, EPOCHS)
    rows = []
    curves = {k: [] for k in NETS}
    for s in SESSIONS:
        idx = np.flatnonzero((d["session"] == s).to_numpy())
        g = d.iloc[idx].reset_index(drop=True)
        mask, y_all = target_rows(g, "up_notup")
        y = np.full(len(g), -1)
        y[np.flatnonzero(mask)] = y_all
        dates = g["date"].to_numpy()
        sym = g["d_sym_advanc"].to_numpy(float)
        tab = g[feats[s]].to_numpy(float)
        tab_no_sym = g[[c for c in feats[s] if c != "d_sym_advanc"]].to_numpy(float)
        sq, im = seqs[idx], small[idx]
        ret = np.column_stack([return_features(sq), sym])
        splits = list(walk_forward_splits(dates, BLOCK, ev["min_train_days"], ev["holdout_days"]))
        for n, (_, te_days) in enumerate(splits[:2] if smoke else splits):
            t0 = time.perf_counter()
            start = np.min(te_days)
            tr, te = (dates < start) & (y >= 0), np.isin(dates, te_days)
            fold = str(start)[:10]
            r = g.loc[te, KEYS].copy()
            r["return_lr"] = proba(build("lr", {"C": TAB_C}, 0).fit(ret[tr], y[tr]), ret[te])
            r["tab_lr"] = proba(build("lr", {"C": TAB_C}, 0).fit(tab[tr], y[tr]), tab[te])
            r["cnn1d_3"] = fit_predict_net("cnn1d_3", sq[:, :3], sym, None, y, tr, te, dates, dev, seeds, epochs, fold, curves["cnn1d_3"])
            r["cnn1d_ext"] = fit_predict_net("cnn1d_ext", sq, sym, None, y, tr, te, dates, dev, seeds, epochs, fold, curves["cnn1d_ext"])
            r["cnn2d_small"] = fit_predict_net("cnn2d_small", im, sym, None, y, tr, te, dates, dev, seeds, epochs, fold, curves["cnn2d_small"])
            r["fusion"] = fit_predict_net("fusion", sq, sym, tab_no_sym, y, tr, te, dates, dev, seeds, epochs, fold, curves["fusion"])
            rows.append(r)
            print(s, fold, f"{n + 1}/{len(splits)}", f"{time.perf_counter() - t0:.0f} วิ", flush=True)
        pd.concat(rows, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    p = pd.concat(rows, ignore_index=True)
    assert p.drop(columns=KEYS).notna().all().all() and p[KEYS].duplicated().sum() == 0
    p.to_csv(out_dir / "predictions.csv", index=False)
    for k, c in curves.items():
        pd.DataFrame(c).to_csv(out_dir / f"{k}_curves.csv", index=False)
    print("ครบ", len(p), "แถว", flush=True)


def report(rules):
    s31, s42 = [importlib.import_module(n) for n in ("31_e14_finetune", "42_overfit_audit")]
    _, d, _, _ = prepare(rules)
    tz = rules["timezone"]
    RES.mkdir(parents=True, exist_ok=True)
    un = s42.e0_wf(d, "up_notup", rules)
    p = s42.load_pred(OUT / "predictions.csv", tz)
    res = s42.load_pred(OUTPUT_DIR / "experiments" / "E20" / "cnnb_resnet18_s2.csv", tz).rename(columns={"p_up": "resnet18_s2"})
    x = d[KEYS + ["direction"]].merge(un[KEYS + ["e0"]], on=KEYS).merge(p, on=KEYS).merge(res[KEYS + ["resnet18_s2"]], on=KEYS)
    x = x[x["direction"].notna()].sort_values(KEYS).reset_index(drop=True)
    labeled = p.merge(d[KEYS + ["direction"]], on=KEYS)["direction"].notna().sum()     # แถวที่ไม่มีป้ายไม่ถูกนับ
    assert len(p) == len(res) and len(x) == labeled, (len(x), labeled, len(p), len(res))
    x = x.rename(columns={m: f"p_{m}" for m in ORDER})
    y = (x["direction"] == 1).to_numpy(float)
    rows, fam = [], {}
    for m in ORDER:
        fam[m] = x[KEYS].assign(ll=s42.ll(y, x[f"p_{m}"]))
        for s in SESSIONS:
            r = {"model": m, "session": s}
            h = x[x["session"] == s]
            yy = (h["direction"] == 1).to_numpy(float)
            r.update(n=len(h), ll=s42.ll(yy, h[f"p_{m}"]).mean(), acc=((h[f"p_{m}"] > .5) == (yy == 1)).mean(), pred_up=(h[f"p_{m}"] > .5).mean())
            if m != "e0":
                r.update(s42.test_gain(x.assign(p=x[f"p_{m}"]), un, s))
            rows.append(r)
        c = OUT / f"{m}_curves.csv"
        if m in NETS and c.exists():
            cr = s42.curve_row(c)
            for r in rows[-2:]:
                r.update({k: cr[k] for k in ("best_epoch_median", "last_epoch_median", "share_best_le3", "train_drop_after")})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "ablation.csv", index=False, encoding="utf-8-sig")
    cmp99 = pd.concat([s31.compare(x, a, b, .997, 2000) for a, b in PAIRS], ignore_index=True)
    cmp95 = pd.concat([s31.compare(x, a, b, .95, 2000) for a, b in PAIRS], ignore_index=True)
    c = cmp99.merge(cmp95[["a", "b", "group", "d_ll_lo", "d_ll_hi", "d_acc_lo", "d_acc_hi"]], on=["a", "b", "group"], suffixes=("", "_95"))
    wins = []
    for (a, b), q in c.groupby(["a", "b"], sort=False):
        q = q.set_index("group")
        for th in ("เช้า", "บ่าย"):
            main = q.loc[th]
            stock_ok = all(q.loc[f"{sym} {th}", "d_ll_lo"] <= 0 for sym in ("KBANK", "ADVANC"))
            wins.append({"a": a, "b": b, "session": th, "d_ll": main["d_ll"], "d_ll_lo": main["d_ll_lo"], "d_ll_hi": main["d_ll_hi"], "d_ll_lo_95": main["d_ll_lo_95"], "d_ll_hi_95": main["d_ll_hi_95"],
                         "d_acc": main["d_acc"], "d_acc_lo": main["d_acc_lo"], "d_acc_hi": main["d_acc_hi"], "win": bool(main["d_ll_hi"] < 0 and main["d_acc_hi"] >= 0 and stock_ok)})
    w = pd.DataFrame(wins)
    c.to_csv(RES / "pairs_detail.csv", index=False, encoding="utf-8-sig")
    w.to_csv(RES / "pairs.csv", index=False, encoding="utf-8-sig")
    b = pd.DataFrame(s42.pbo_row("E21", fam))
    b.to_csv(RES / "pbo.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))
    print(w.round(4).to_string(index=False))
    print(b.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=[])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    rules = load_rules()
    if args.smoke:
        run(rules, smoke=True)
        print("** SMOKE: ตรวจ pipeline เท่านั้น ห้ามอ้างเป็นผล **")
    if "run" in args.part:
        run(rules)
    if "report" in args.part:
        report(rules)


if __name__ == "__main__":
    main()
