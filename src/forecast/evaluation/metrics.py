"""ตัวชี้วัด: ทิศทาง, probability, ราคา (บาท / ช่องราคา), ช่วงราคา และเสถียรภาพ"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# ช่องราคา (tick) ของ SET ตามช่วงราคา
TICK_TABLE = [(2, 0.01), (5, 0.02), (10, 0.05), (25, 0.10), (100, 0.25),
              (200, 0.50), (400, 1.00), (np.inf, 2.00)]


def tick_size(price):
    price = np.asarray(price, float)
    return np.select([price < hi for hi, _ in TICK_TABLE], [t for _, t in TICK_TABLE], np.nan)


def round_to_tick(price):
    t = tick_size(price)
    return np.round(np.asarray(price, float) / t) * t


def price_quantiles(df, quantiles):
    """แปลง quantile ของ z กลับเป็นราคา: P(c) × exp(ẑ) — median แปลงกลับเป็น median ได้ตรง"""
    out = pd.DataFrame(index=df.index)
    for q in quantiles:
        c = f"q{round(q * 100):02d}"
        out[f"price_{c}"] = df["p_ref"] * np.exp(df[c])
    return out


def row_losses(df, quantiles):
    y = df["up"].to_numpy(float)
    p = df["p_up"].clip(1e-4, 1 - 1e-4).to_numpy(float)
    z = df["z"].to_numpy(float)
    pred_price = df["p_ref"] * np.exp(df["q50"])
    out = pd.DataFrame(index=df.index)
    out["correct"] = ((p >= 0.5) == (y == 1)).astype(float)
    out["logloss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    # ตัวชี้วัดหลักรอบที่ 2: อาจารย์ให้คะแนน "ขึ้นหรือลง" → วัน flat ไม่มีคำตอบที่ถูก
    #   err_nf = ผิดทิศ เฉพาะวันไม่ flat (NaN ในวัน flat), err_fw = ผิดทิศ โดย flat นับผิดเสมอ
    #   logloss_ud = log loss ของ P(ขึ้น) เทียบ ขึ้น vs ลง เฉพาะวันไม่ flat
    d = df["direction"].to_numpy(float)
    nf = np.isfinite(d) & (d != 0)
    wrong = (np.where(p >= 0.5, 1.0, -1.0) != d).astype(float)
    out["err_nf"] = np.where(nf, wrong, np.nan)
    out["err_fw"] = wrong
    yud = (d == 1).astype(float)
    out["logloss_ud"] = np.where(nf, -(yud * np.log(p) + (1 - yud) * np.log(1 - p)), np.nan)
    out["brier"] = (p - y) ** 2
    out["abs_err_baht"] = (df["p_end"] - pred_price).abs()
    out["abs_err_ticks"] = out["abs_err_baht"] / tick_size(df["p_ref"])
    out["abs_err_naive_baht"] = (df["p_end"] - df["p_ref"]).abs()
    pin = []
    for q in quantiles:
        e = z - df[f"q{round(q * 100):02d}"].to_numpy(float)
        pin.append(np.maximum(q * e, (q - 1) * e))
    out["pinball"] = np.mean(pin, axis=0)
    out["cov50"] = ((z >= df["q25"]) & (z <= df["q75"])).astype(float)
    out["cov80"] = ((z >= df["q10"]) & (z <= df["q90"])).astype(float)
    out["width50_baht"] = df["p_ref"] * (np.exp(df["q75"]) - np.exp(df["q25"]))
    out["width80_baht"] = df["p_ref"] * (np.exp(df["q90"]) - np.exp(df["q10"]))
    labeled = df["up"].notna() & df["z"].notna()
    return out[labeled]


def summarize(df, quantiles, by=("symbol", "session")):
    losses = row_losses(df, quantiles)
    d = df.loc[losses.index]

    def one(idx):
        g, l = d.loc[idx], losses.loc[idx]
        nonflat = g["direction"] != 0
        pred_dir = np.where(g["p_up"] >= 0.5, 1, -1)
        y = g["up"].astype(int)
        return pd.Series({
            "n": len(g),
            "n_flat": int((~nonflat).sum()),
            "up_rate": y.mean(),
            # ขึ้น vs ไม่ขึ้น (เป้าหมายที่โมเดลฝึก)
            "accuracy_up": l["correct"].mean(),
            # ทายขึ้น/ลง เทียบ label สามสถานะ — รายงานทั้งสองวิธีจนกว่าอาจารย์ยืนยัน (rules.yaml → label.flat.scoring)
            "acc_flat_as_wrong": (pred_dir == g["direction"]).mean(),
            "acc_flat_excluded": (pred_dir[nonflat] == g.loc[nonflat, "direction"]).mean(),
            "balanced_accuracy": balanced_accuracy_score(y, g["p_up"] >= 0.5) if y.nunique() > 1 else np.nan,
            "logloss": l["logloss"].mean(),
            "logloss_ud": l["logloss_ud"].mean(),
            "brier": l["brier"].mean(),
            "mae_baht": l["abs_err_baht"].mean(),
            "mae_ticks": l["abs_err_ticks"].mean(),
            "mae_naive_baht": l["abs_err_naive_baht"].mean(),
            "coverage50": l["cov50"].mean(),
            "coverage80": l["cov80"].mean(),
            "width50_baht": l["width50_baht"].mean(),
            "width80_baht": l["width80_baht"].mean(),
            "pinball": l["pinball"].mean(),
        })

    rows = [one(d.index).rename("ALL")]
    for key, g in d.groupby(list(by)):
        rows.append(one(g.index).rename("/".join(map(str, np.atleast_1d(key)))))
    return pd.DataFrame(rows).rename_axis("group").reset_index()
