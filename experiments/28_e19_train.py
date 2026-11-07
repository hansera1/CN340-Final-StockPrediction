"""E19: CNN บนภาพ 1m (ดัดแปลงจาก 26_train_1m_cnn.py ของเพื่อน ตามแผนรอบ 7 ข้อ 10) — งานสำรวจ ไม่ใช้ตัดสิน

    python experiments/28_e19_train.py               # ขั้นต่ำฝึก 20 วัน · วันทดสอบ < 5 → รายงานว่าวัดผลไม่ได้
    python experiments/28_e19_train.py --smoke 8     # ตรวจ pipeline เท่านั้น: ฝึกขั้นต่ำ 8 วัน, 2 epoch, 1 seed · ห้ามอ้างเป็นผล
    python experiments/28_e19_train.py --explore 8   # รันสำรวจ: ฝึกขั้นต่ำ 8 วัน สเปกเต็ม · ไม่ผ่านเกณฑ์ที่ลงทะเบียน ไม่ใช้ตัดสิน
→ outputs/experiments/E19[_smoke]/predictions.csv, curves.csv, metrics.csv
"""
import argparse
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.config import OUTPUT_DIR, PROCESSED_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import design_matrix, load_or_build_dataset
from forecast.models import get_model
from forecast.models.e9 import last_days_split
from forecast.models.finetune import CURVES, MEAN, STD, FineTuneModel
from forecast.models.stage2 import Stage2

warnings.filterwarnings("ignore")
ARCHS = ["resnet18", "vgg16_bn", "densenet121"]
MIN_TRAIN, MIN_TEST = 20, 5
CS = (0.003, 0.01, 0.03, 0.1)
STAGE2_LAM = 45.0
FT_KW = {"patience": 5}


def extractor(arch, dev):
    """เวกเตอร์ของ backbone: ResNet18 512 · VGG16-BN fc7 4096 · DenseNet121 1024"""
    import torchvision.models as tv
    if arch == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
    elif arch == "vgg16_bn":
        m = tv.vgg16_bn(weights=tv.VGG16_BN_Weights.IMAGENET1K_V1)
        m.classifier = m.classifier[:-1]
    else:
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = torch.nn.Identity()
    return m.eval().to(dev)


@torch.no_grad()
def embed(arch, imgs, dev, batch=64):
    net = extractor(arch, dev)
    out = []
    x = torch.from_numpy(np.ascontiguousarray(imgs))
    for s in range(0, len(x), batch):
        b = x[s:s + batch].to(dev).float().div_(255)
        out.append(net((b - MEAN.to(dev)) / STD.to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def pca_features(emb, tr, n):
    """PCA fit บนข้อมูลฝึกเท่านั้น → ทุกแถว"""
    n = min(n, int(tr.sum()) - 1, emb.shape[1])
    return PCA(n_components=n, random_state=0).fit(emb[tr]).transform(emb)


def a_un_oos(g, cols, rules, days):
    """P(ขึ้น) นอกตัวอย่างของ A-un: ฝึกจากวันก่อนวันนั้นเท่านั้น → Series ตาม index ของ g"""
    d = g["date"].to_numpy()
    X = design_matrix(g, cols, ["session"])
    out = pd.Series(np.nan, index=g.index)
    for day in days:
        a, b = (d < day) & g["direction"].notna().to_numpy(), d == day
        if a.sum() < 120:
            continue
        m = get_model("e12_lr", rules["quantiles"], target="up_notup", permutation=False, n_bags=5).fit(X[a], g["up"].to_numpy()[a], g["z"].to_numpy()[a], g[a])
        out[b] = m.predict(X[b], g[b])["p_up"].to_numpy()
    return out


def fit_predict(kind, arch, emb, imgs, g, y, tr, te, p1, rules, epochs, seeds):
    """Model A หนึ่งแบบ → P(ขึ้น) ของแถว te · ใช้ข้อมูล tr เท่านั้น"""
    dates = g["date"].to_numpy()
    if kind in ("S0-pca3", "S0-pca5"):
        F = pca_features(emb, tr, int(kind[-1]))
        return Stage2(STAGE2_LAM).fit(F[tr], y[tr], p1[tr]).predict(F[te], p1[te])
    if kind == "S0-LR":
        a, b = last_days_split(dates[tr], 0.25)
        scores = {}
        X, yy = emb[tr], y[tr]
        for C in CS:
            if len(np.unique(yy[a])) == 2 and b.any():
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[a], yy[a])
                scores[C] = log_loss(yy[b], m.predict_proba(X[b])[:, 1], labels=[0, 1])
        C = min(scores, key=scores.get) if scores else 0.01
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X, yy).predict_proba(emb[te])[:, 1]
    stage = kind.lower()
    m = FineTuneModel(rules["quantiles"], arch=arch, stage=stage, seeds=seeds, epochs=epochs, target="up_notup", **FT_KW)
    idx = g.index
    m.fit(pd.DataFrame(index=idx[tr]), g["up"].to_numpy()[tr], g["z"].to_numpy()[tr], g[tr], imgs[tr])
    return m.predict(pd.DataFrame(index=idx[te]), g[te], imgs[te])["p_up"].to_numpy()


def oof_days(dates, tr, n=4):
    """fold ตามวันภายในข้อมูลฝึก: แต่ละ fold ฝึกจากวันอื่น ทำนายวันใน fold · วันเดียวกันอยู่ fold เดียว"""
    ud = np.unique(dates[tr])
    return [np.isin(dates, part) & tr for part in np.array_split(ud, min(n, len(ud)))]


def day_ci(v, dates, n=2000, seed=0):
    """ผลต่างเฉลี่ย + CI 95% จาก bootstrap รายวัน"""
    s = pd.DataFrame({"v": v, "d": dates}).groupby("d")["v"].agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    b = s["sum"].to_numpy()[idx].sum(1) / s["count"].to_numpy()[idx].sum(1)
    return s["sum"].sum() / s["count"].sum(), np.quantile(b, .025), np.quantile(b, .975)


def explore_ci(r):
    """เทียบ E0-un และ A-un บนแถวเดียวกัน → ตารางรายแบบ"""
    y = (r["direction"] == 1).to_numpy(float)
    ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    ok = lambda p: ((p > .5) == (y == 1)).astype(float)
    out = {}
    for (arch, kind), idx in r.groupby(["arch", "kind"]).groups.items():
        m = r.index.isin(idx)
        row = {}
        p = r["p_up"].to_numpy()
        for name, c in [("e0", "p_e0"), ("a_un", "p_a_un")]:
            q = r[c].to_numpy(float)
            row[f"acc_{name}"] = ok(q)[m].mean()
            row[f"d_acc_{name}"], row[f"d_acc_{name}_lo"], row[f"d_acc_{name}_hi"] = day_ci((ok(p) - ok(q))[m], r["date"].to_numpy()[m])
            row[f"d_ll_{name}"], row[f"d_ll_{name}_lo"], row[f"d_ll_{name}_hi"] = day_ci((ll(p) - ll(q))[m], r["date"].to_numpy()[m])
        out[(arch, kind)] = row
    return pd.DataFrame(out).T.rename_axis(["arch", "kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, help="ฝึกขั้นต่ำ n วัน เพื่อตรวจ pipeline เท่านั้น")
    ap.add_argument("--explore", type=int, help="ฝึกขั้นต่ำ n วัน สเปกเต็ม — รันสำรวจ ไม่ผ่านเกณฑ์ที่ลงทะเบียน")
    ap.add_argument("--minutes", type=int, default=1, choices=[1, 5], help="ภาพจากแท่ง 5 นาที: e19_images_5m.npz (สร้างด้วย 27_e19_1m_scalogram.py --minutes 5) · ใช้คู่กับ --explore")
    ap.add_argument("--strong", action="store_true", help="E20: regularization แรงกว่าเดิม ใช้คู่กับ --explore")
    args = ap.parse_args()
    smoke, explore = args.smoke is not None, args.explore is not None
    suffix = "" if args.minutes == 1 else f"_{args.minutes}m"
    img_suffix = suffix
    if args.strong:      # E20: results/e20_preregistration.md
        global CS, STAGE2_LAM, FT_KW
        CS, STAGE2_LAM = (0.0003, 0.001), 200.0
        FT_KW = {"patience": 2, "lr_head": 3e-4, "lr_block": 3e-5, "weight_decay": 0.2}
        suffix += "_strong"
    min_train, epochs, seeds = (args.smoke, 2, (0,)) if smoke else (args.explore if explore else MIN_TRAIN, 30, (0, 1, 2))
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rules = load_rules()
    assert rules["label"]["price_1630"] == "closing_auction"
    ds = load_or_build_dataset(rules, tag="_e19", rebuild=True)
    z = np.load(PROCESSED_DIR / f"e19_images{img_suffix}.npz", allow_pickle=False)
    pos = {k: i for i, k in enumerate(z["keys"].tolist())}
    pm = ds[ds["session"] == "afternoon"].reset_index(drop=True)
    key = pm["symbol"] + "|" + pm["date"].dt.strftime("%Y-%m-%d")
    cols = next(v for v in load_experiments()["E12"]["variants"] if v["name"] == "fd_lr")["features"]["afternoon"]
    keep = key.isin(pos) & pm["direction"].notna() & pm["p_end"].notna()
    p1_all = a_un_oos(pm, cols, rules, np.unique(pm.loc[keep, "date"].to_numpy()))
    g = pm[keep].sort_values(["date", "symbol"]).reset_index(drop=True)
    p1 = p1_all[keep.to_numpy()].reindex(pm[keep].sort_values(["date", "symbol"]).index).to_numpy(float)
    imgs = z["img"][[pos[k] for k in (g["symbol"] + "|" + g["date"].dt.strftime("%Y-%m-%d"))]]
    y = (g["direction"] == 1).to_numpy(int)
    dates = g["date"].to_numpy()
    ud = np.unique(dates)
    test_days = ud[min_train:]
    msg = f"{len(g)} ตัวอย่าง · {len(ud)} วัน · ขั้นต่ำฝึก {min_train} · วันทดสอบ {len(test_days)}"
    print(msg, flush=True)
    if not smoke and not explore and len(test_days) < MIN_TEST:
        print(f"วัดผลไม่ได้: วันทดสอบ {len(test_days)} < {MIN_TEST} · ต้องมีอย่างน้อย {MIN_TRAIN + MIN_TEST} วัน")
        return
    out = OUTPUT_DIR / "experiments" / (("E19_smoke" if smoke else "E19_explore" if explore else "E19") + suffix)
    hist = pm[pm["direction"].notna()]
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["S0-pca3", "S0-pca5", "S0-LR", "S1", "S2"]
    rows, times = [], []
    for arch in ARCHS:
        emb = embed(arch, imgs, dev)
        for kind in kinds:
            t0 = time.perf_counter()
            for day in test_days:
                tr, te = dates < day, dates == day
                p = fit_predict(kind, arch, emb, imgs, g, y, tr, te, p1, rules, epochs, seeds)
                oof = np.full(len(g), np.nan)
                for f in oof_days(dates, tr):
                    ftr = tr & ~f
                    if len(np.unique(y[ftr])) == 2:
                        oof[f] = fit_predict(kind, arch, emb, imgs, g, y, ftr, f, p1, rules, epochs, seeds)
                ok = tr & np.isfinite(oof)
                zhat = LinearRegression().fit(oof[ok, None], g["z"].to_numpy()[ok]).predict(p[:, None])
                price = g["p_ref"].to_numpy()[te] * np.exp(zhat)
                e0 = [(hist[(hist["date"].to_numpy() < day) & (hist["symbol"] == s)]["direction"] == 1).mean() for s in g.loc[te, "symbol"]]
                rows.append(g.loc[te, ["date", "symbol", "direction", "p_ref", "p_end"]].assign(arch=arch, kind=kind, p_up=p, price=price, p_a_un=p1[te], p_e0=e0))
            times.append({"arch": arch, "kind": kind, "sec": time.perf_counter() - t0})
            print(arch, kind, f"{times[-1]['sec']:.0f} วิ", flush=True)
    r = pd.concat(rows, ignore_index=True)
    r.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(CURVES).to_csv(out / "curves.csv", index=False)
    pd.DataFrame(times).to_csv(out / "times.csv", index=False)
    yy = (r["direction"] == 1).astype(float)
    r["ok"] = (r["p_up"] > .5) == (yy == 1)
    r["ll"] = -(yy * np.log(r["p_up"].clip(1e-6)) + (1 - yy) * np.log((1 - r["p_up"]).clip(1e-6)))
    r["ae"], r["se"], r["ae_pc"], r["se_pc"] = (r["p_end"] - r["price"]).abs(), (r["p_end"] - r["price"]) ** 2, (r["p_end"] - r["p_ref"]).abs(), (r["p_end"] - r["p_ref"]) ** 2
    t = r.groupby(["arch", "kind"]).agg(n=("ok", "size"), acc=("ok", "mean"), ll=("ll", "mean"), up_rate=("direction", lambda s: (s == 1).mean()), pred_up=("p_up", lambda s: (s > .5).mean()),
                                        mae=("ae", "mean"), rmse=("se", lambda s: np.sqrt(s.mean())), mae_pc=("ae_pc", "mean"), rmse_pc=("se_pc", lambda s: np.sqrt(s.mean()))).reset_index()
    t.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    if smoke:
        print("** SMOKE: ตรวจ pipeline เท่านั้น ห้ามอ้างเป็นผล **")
    elif explore:
        t = t.join(explore_ci(r), on=["arch", "kind"])
        t.to_csv(ROOT / "results" / "e19" / f"explore_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
        print("** EXPLORE: ข้อมูลน้อยเกินสรุป ไม่ผ่านเกณฑ์ที่ลงทะเบียน ไม่ใช้ตัดสิน **")
    else:
        t.to_csv(ROOT / "results" / "e19" / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
