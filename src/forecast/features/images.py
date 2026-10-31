"""ภาพสำหรับ E7 (ResNet18 pretrained) จากหน้าต่างแท่งที่จบแล้ว ณ จุดตัด — ข้ามวันได้ (60m: 56 แท่ง ≈ 8 วัน)

- E7a scalogram: Morlet CWT ของ r, |r|, log volume → 3 ช่องของภาพ (แกนตั้ง = scale, แกนนอน = เวลา)
- E7b กราฟแท่งเทียน + volume: ช่อง R = แท่งขึ้น, G = แท่งลง, B = volume (แบบ Jiang–Kelly–Xiu)

ทุกภาพใช้เฉพาะแท่งที่ bar_end ≤ จุดตัด ไม่มี flip / rotate / crop
"""
import numpy as np
import pandas as pd

from forecast.config import session_times

SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def windows(bars, rows, rules, length):
    """แท่ง `length` แท่งล่าสุดที่จบ ณ/ก่อนจุดตัดของแต่ละแถว → list ของ DataFrame (อาจสั้นกว่า length)"""
    b = bars.sort_values("bar_end").reset_index(drop=True)
    ends = pd.DatetimeIndex(b["bar_end"])
    out = []
    for date, sess in rows[["date", "session"]].itertuples(index=False):
        cutoff = session_times(rules, sess, date)["cutoff"]
        i = int(ends.searchsorted(cutoff, side="right"))
        out.append(b.iloc[max(0, i - length):i])
    return out


def _series(w):
    c = np.log(w["close"].to_numpy(float))
    r = np.diff(c, prepend=c[0]) * 100
    v = np.log1p(w["volume"].to_numpy(float))
    v = (v - v.mean()) / (v.std() + 1e-8)
    return np.stack([r, np.abs(r), v])


def morlet_cwt(x, n_scales=32, w0=6.0):
    """|CWT| ด้วย Morlet (คำนวณเอง — scipy.signal.cwt ถูกถอดออกแล้ว) → (n_scales, len(x))"""
    n = len(x)
    scales = np.geomspace(1, max(2, n / 2), n_scales)
    out = np.empty((n_scales, n))
    for k, s in enumerate(scales):
        half = min(int(4 * s), (n - 1) // 2)          # kernel ไม่ยาวกว่าสัญญาณ → ผลยาว n เท่าเดิม
        t = np.arange(-half, half + 1)
        psi = np.exp(1j * w0 * t / s) * np.exp(-0.5 * (t / s) ** 2) / np.sqrt(s)
        out[k] = np.abs(np.convolve(x, np.conj(psi[::-1]), mode="same"))
    return out


def _resize(img, h=SIZE, w=SIZE):
    """nearest-neighbour resize (C, H, W) → (C, h, w)"""
    C, H, W = img.shape
    return img[:, (np.arange(h) * H // h)[:, None], (np.arange(w) * W // w)[None, :]]


def _unit(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def scalogram_image(w):
    x = _series(w)
    chans = [_unit(np.log1p(morlet_cwt((c - c.mean()) / (c.std() + 1e-8)))) for c in x]
    return _resize(np.stack(chans).astype(np.float32))


def candlestick_image(w, length):
    img = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    px = SIZE // length                       # ความกว้างต่อแท่ง (60m, 56 แท่ง → 4 px)
    price_h, vol_top = int(SIZE * 0.75), int(SIZE * 0.80)
    o = w["open"].to_numpy(float) if "open" in w else w["close"].to_numpy(float)
    h = w["high"].to_numpy(float) if "high" in w else np.maximum(o, w["close"].to_numpy(float))
    l = w["low"].to_numpy(float) if "low" in w else np.minimum(o, w["close"].to_numpy(float))
    c, v = w["close"].to_numpy(float), w["volume"].to_numpy(float)
    lo, hi = l.min(), h.max()
    y = lambda p: (price_h - 1 - np.round((p - lo) / (hi - lo + 1e-12) * (price_h - 1))).astype(int)
    vmax = v.max() if v.max() > 0 else 1.0
    start = length - len(w)                   # หน้าต่างสั้นชิดขวา (แท่งล่าสุดอยู่ขวาสุดเสมอ)
    for i in range(len(w)):
        x0 = (start + i) * px
        ch = 0 if c[i] >= o[i] else 1
        img[ch, y(h[i]):y(l[i]) + 1, x0 + px // 2 - 1 if px > 2 else x0] = 1.0       # ไส้เทียน
        top, bot = sorted((y(o[i]), y(c[i])))
        img[ch, top:bot + 1, x0:x0 + max(1, px - 1)] = 1.0                           # ตัวเทียน
        vh = int(round(v[i] / vmax * (SIZE - vol_top)))
        if vh:
            img[2, SIZE - vh:, x0:x0 + max(1, px - 1)] = 1.0
    return img


def build_images(bars, rows, rules, kind, length=56, normalize=True):
    """→ (n, 3, 224, 224) float32 normalize แบบ ImageNet (normalize=False: ค่า 0–1 ไว้เก็บเป็น uint8);
    แถวที่ไม่มีแท่งเลยเป็นภาพว่าง"""
    out = np.zeros((len(rows), 3, SIZE, SIZE), dtype=np.float32)
    for i, w in enumerate(windows(bars, rows, rules, length)):
        if len(w) >= 4:
            out[i] = scalogram_image(w) if kind == "scalogram" else candlestick_image(w, length)
    return (out - IMAGENET_MEAN) / IMAGENET_STD if normalize else out
