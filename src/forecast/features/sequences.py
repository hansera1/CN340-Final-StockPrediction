"""ลำดับข้อมูลสำหรับ 1D-CNN (E4): signed return, |r|, log volume ของแท่งที่จบก่อนจุดตัด

รอบบ่าย = แท่งวันนี้ถึงจุดตัด (C1); รอบเช้า = แท่งของวันก่อน (C2)
"""
import numpy as np

from forecast.config import session_times
from forecast.data.asof import bars_until

N_CHANNELS = 3


def build_sequences(bars, rows, rules, length=120):
    out = np.zeros((len(rows), N_CHANNELS, length), dtype=np.float32)
    groups = dict(tuple(bars.groupby("date", sort=True)))
    dates = sorted(groups)
    for i, (date, sess) in enumerate(rows[["date", "session"]].itertuples(index=False)):
        cutoff = session_times(rules, sess, date)["cutoff"]
        b = bars_until(groups[date], cutoff) if date in groups else None
        if b is None or len(b) < 2:
            prev = [d for d in dates if d < date]
            if not prev:
                continue
            b = groups[prev[-1]]
        lr = np.diff(np.log(b["close"].to_numpy(dtype=float))) * 100
        lv = np.log1p(b["volume"].to_numpy(dtype=float))[1:]
        lv = (lv - lv.mean()) / (lv.std() + 1e-8)
        seq = np.stack([lr, np.abs(lr), lv])[:, -length:]
        out[i, :, length - seq.shape[1]:] = np.nan_to_num(seq)
    return out
