"""ROC curve + confusion matrix ของ logistic regression ที่ล็อก (E12 fd_lr) — อ่านไฟล์ผลทำนายที่มีอยู่ ไม่ฝึกหรือรันโมเดลใหม่

- ผล walk-forward 599 วันทดสอบ (outputs/experiments/E12/*_predictions.csv) = ชุดที่ใช้ตัดสิน
- holdout 10 วัน (outputs/experiments/E12_holdout/) รันไปแล้วครั้งเดียวในรอบที่ 3 — แสดงแยกเพื่อรายงานเท่านั้น ไม่รันซ้ำ
- เป้าหมายขึ้น vs ลงบนรอบที่ไม่ flat · ทายขึ้นเมื่อ P(ขึ้น) ≥ 0.5 · AUC CI = block bootstrap รายวัน 2,000 รอบ
- เทียบ LR ทุกตัว (E9 lr / lrx, E12 f0_lr / fd_lr / fdh_lr) และ E0-ud บนแถวเดียวกัน

    python experiments/25_roc_confusion.py
→ results/roc/roc_confusion.png, roc_by_group.png, metrics.csv, confusion.csv
"""
import _setup  # noqa: F401
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from forecast.config import OUTPUT_DIR, ROOT  # noqa: E402

plt.rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "DejaVu Sans"]
EXP = OUTPUT_DIR / "experiments"
OUT = ROOT / "results" / "roc"
KEYS = ["date", "symbol", "session"]
LR = {"E9 lr": "E9/lr", "E9 lrx": "E9/lrx", "E12 f0_lr": "E12/f0_lr", "E12 fd_lr": "E12/fd_lr",
      "E12 fdh_lr": "E12/fdh_lr"}
BEST = "E12 fd_lr"
GROUPS = [("KBANK", "morning"), ("KBANK", "afternoon"), ("ADVANC", "morning"), ("ADVANC", "afternoon")]
TH = {"morning": "เช้า", "afternoon": "บ่าย"}
N_BOOT = 2000


def load(rel):
    p = pd.read_csv(EXP / f"{rel}_predictions.csv", usecols=KEYS + ["direction", "p_up"])
    return p[p["direction"].notna()].set_index(KEYS).sort_index()   # ไม่มี label (ข้อมูลปลายรอบขาด) → ไม่นับ


def boot_auc(d, rng):
    """CI ของ AUC: สุ่มวันแบบมีคืน (ทุกแถวของวันไปด้วยกัน)"""
    days = d["date"].unique()
    idx = {k: v for k, v in d.groupby("date").indices.items()}
    y, s, out = d["y"].to_numpy(), d["p_up"].to_numpy(), []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[k] for k in rng.choice(days, len(days))])
        if len(np.unique(y[rows])) == 2:
            out.append(roc_auc_score(y[rows], s[rows]))
    return np.percentile(out, [2.5, 97.5])


def metrics(name, d, rng, boot=True):
    y, s = d["y"].to_numpy(), d["p_up"].to_numpy()
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    lo, hi = boot_auc(d, rng) if boot else (np.nan, np.nan)
    eps = 1e-12
    return {"model": name, "n_nonflat": len(d), "up_rate": y.mean(), "acc": (tp + tn) / len(d),
            "auc": roc_auc_score(y, s), "auc_lo": lo, "auc_hi": hi,
            "logloss_ud": -np.mean(y * np.log(s + eps) + (1 - y) * np.log(1 - s + eps)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision_up": tp / (tp + fp) if tp + fp else np.nan, "recall_up": tp / (tp + fn),
            "precision_down": tn / (tn + fn) if tn + fn else np.nan, "recall_down": tn / (tn + fp),
            "pred_up_rate": pred.mean()}


def nonflat(p):
    d = p[p["direction"] != 0].reset_index()
    d["y"] = (d["direction"] == 1).astype(int)
    return d


def draw_cm(ax, d, title):
    cm = confusion_matrix(d["y"], (d["p_up"] >= 0.5).astype(int), labels=[1, 0])
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({v / cm[i].sum():.1%} ของแถว)", ha="center", va="center", fontsize=10,
                    color="white" if v > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1], ["ทาย ขึ้น", "ทาย ลง"]); ax.set_yticks([0, 1], ["จริง ขึ้น", "จริง ลง"])
    ax.set_title(title, fontsize=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    preds = {k: load(v) for k, v in LR.items()}
    preds["E0-ud (baseline)"] = load("E12/baseline_ud")
    common = None
    for p in preds.values():
        common = p.index if common is None else common.intersection(p.index)
    preds = {k: p.loc[common] for k, p in preds.items()}
    assert all((p["direction"].values == preds[BEST]["direction"].values).all() for p in preds.values())
    nf = {k: nonflat(p) for k, p in preds.items()}

    rows = [metrics(k, d, rng) for k, d in nf.items()]
    best = nf[BEST]
    for sym, sess in GROUPS:
        g = best[(best.symbol == sym) & (best.session == sess)]
        rows.append(metrics(f"{BEST} · {sym} {sess}", g, rng))
    ho = nonflat(load("E12_holdout/fd_lr"))
    rows.append(metrics(f"{BEST} · holdout (รันครั้งเดียวรอบ 3)", ho, rng))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")

    # flat แยกให้เห็น: จริง ขึ้น/flat/ลง × ทาย ขึ้น/ลง (ไฟล์ส่งทายได้แค่ขึ้นหรือลง)
    full = preds[BEST].reset_index()
    full["actual"] = full["direction"].map({1.0: "ขึ้น", 0.0: "flat", -1.0: "ลง"})
    full["pred"] = np.where(full["p_up"] >= 0.5, "ทาย ขึ้น", "ทาย ลง")
    cm3 = pd.crosstab(full["actual"], full["pred"]).reindex(["ขึ้น", "flat", "ลง"])
    cm3.to_csv(OUT / "confusion.csv", encoding="utf-8-sig")

    # รูป 1: ROC เทียบทุกตัว + confusion matrix ของ fd_lr (walk-forward / holdout)
    fig = plt.figure(figsize=(17, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    style = {BEST: dict(lw=2.6, color="#c0392b"), "E0-ud (baseline)": dict(lw=1.5, color="0.4", ls="--")}
    for k, d in nf.items():
        fpr, tpr, _ = roc_curve(d["y"], d["p_up"])
        r = t[t.model == k].iloc[0]
        ax.plot(fpr, tpr, label=f"{k}: AUC {r.auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]",
                **style.get(k, dict(lw=1, alpha=0.8)))
    fpr, tpr, thr = roc_curve(best["y"], best["p_up"])
    i = np.argmin(np.abs(thr - 0.5))
    ax.plot(fpr[i], tpr[i], "o", color="#c0392b", ms=8, label="fd_lr ที่เกณฑ์ P ≥ 0.5")
    ax.plot([0, 1], [0, 1], color="0.75", lw=1)
    ax.set_xlabel("False positive rate (ทายขึ้น แต่จริงลง / ลงทั้งหมด)")
    ax.set_ylabel("True positive rate (ทายขึ้นถูก / ขึ้นทั้งหมด)")
    ax.set_title(f"ROC · walk-forward {best['date'].nunique()} วัน · {len(best)} รอบที่ไม่ flat", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right"); ax.set_aspect("equal")

    rb = t[t.model == BEST].iloc[0]
    draw_cm(fig.add_subplot(gs[1]), best, f"fd_lr walk-forward (ไม่นับ flat)\nแม่น {rb.acc:.1%} · n = {len(best)}")
    ax = fig.add_subplot(gs[2])
    ax.imshow(cm3.to_numpy(), cmap="Greens", vmin=0)
    for i_, a in enumerate(cm3.index):
        for j, b in enumerate(cm3.columns):
            ax.text(j, i_, f"{cm3.loc[a, b]}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(2), cm3.columns); ax.set_yticks(range(3), [f"จริง {a}" for a in cm3.index])
    n_fw = len(full)
    acc_fw = ((full["pred"] == "ทาย ขึ้น") & (full["actual"] == "ขึ้น")).sum() + \
             ((full["pred"] == "ทาย ลง") & (full["actual"] == "ลง")).sum()
    ax.set_title(f"fd_lr walk-forward รวม flat\nflat นับเป็นผิด: แม่น {acc_fw / n_fw:.1%} · n = {n_fw}", fontsize=10)
    rh = t[t.model.str.contains("holdout")].iloc[0]
    draw_cm(fig.add_subplot(gs[3]), ho, f"fd_lr holdout 10 วัน (รายงานเท่านั้น)\nแม่น {rh.acc:.1%} · n = {len(ho)}")
    fig.suptitle("Logistic regression ที่ล็อก (E12 fd_lr): ROC และ confusion matrix จากผลทำนายจริง · ทายขึ้นเมื่อ P(ขึ้น) ≥ 0.5",
                 fontsize=12)
    fig.savefig(OUT / "roc_confusion.png", dpi=120, bbox_inches="tight")

    # รูป 2: แยกหุ้น/รอบ — ROC + confusion matrix
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.6))
    for j, (sym, sess) in enumerate(GROUPS):
        g = best[(best.symbol == sym) & (best.session == sess)]
        b0 = nf["E0-ud (baseline)"]
        b0 = b0[(b0.symbol == sym) & (b0.session == sess)]
        r = t[t.model == f"{BEST} · {sym} {sess}"].iloc[0]
        ax = axs[0, j]
        for d, kw, lab in [(g, dict(color="#c0392b", lw=2), f"fd_lr AUC {r.auc:.3f} [{r.auc_lo:.2f}, {r.auc_hi:.2f}]"),
                           (b0, dict(color="0.4", ls="--", lw=1.2), f"E0-ud AUC {roc_auc_score(b0.y, b0.p_up):.3f}")]:
            fpr, tpr, _ = roc_curve(d["y"], d["p_up"])
            ax.plot(fpr, tpr, label=lab, **kw)
        ax.plot([0, 1], [0, 1], color="0.75", lw=1)
        ax.set_title(f"{sym} {TH[sess]} · n = {len(g)} · ขึ้น {g.y.mean():.0%}", fontsize=10)
        ax.legend(fontsize=8, loc="lower right"); ax.set_aspect("equal")
        ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
        draw_cm(axs[1, j], g, f"แม่น {r.acc:.1%} · ทายขึ้น {r.pred_up_rate:.0%} ของรอบ")
    fig.suptitle("E12 fd_lr แยกหุ้น/รอบ (walk-forward, ไม่นับ flat)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "roc_by_group.png", dpi=120, bbox_inches="tight")

    cols = ["model", "n_nonflat", "up_rate", "acc", "auc", "auc_lo", "auc_hi", "logloss_ud", "tp", "fp", "tn", "fn",
            "precision_up", "recall_up", "precision_down", "recall_down", "pred_up_rate"]
    print(t[cols].round(4).to_string(index=False))
    print("\nรวม flat (fd_lr walk-forward):\n", cm3)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
