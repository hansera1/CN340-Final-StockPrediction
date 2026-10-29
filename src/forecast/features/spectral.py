"""ตัวแปรความถี่ (E5 / C8 ย่อ): band power จาก Welch ของ log return รายแท่ง

คำนวณจากข้อมูลที่ตัดตามเวลาแล้วเท่านั้น และไม่ใช้ two-sided filtering
power ของ r กับ –r เท่ากัน จึงต้องใช้คู่กับ signed returns เดิมเสมอ
"""
import numpy as np
from scipy.signal import welch

# คาบ (หน่วย: แท่ง) ของแต่ละย่าน
BANDS = ((2, 5), (5, 15), (15, 60))


def band_powers(close, nperseg=32):
    lr = np.diff(np.log(np.asarray(close, dtype=float)))
    lr = lr[np.isfinite(lr)]
    if len(lr) < nperseg:
        return {}
    freq, pxx = welch(lr - lr.mean(), fs=1.0, nperseg=nperseg)
    total = pxx[1:].sum()
    out = {}
    for lo, hi in BANDS:
        mask = (freq >= 1 / hi) & (freq < 1 / lo)
        out[f"f_bp_{lo}_{hi}"] = float(pxx[mask].sum() / total) if total > 0 else np.nan
    return out
