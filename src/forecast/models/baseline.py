"""แบบ A / E0: ราคา = ราคาล่าสุดที่รู้, ทิศทาง = ความถี่ "ขึ้น" ในอดีต

target="updown" (รอบที่ 2): P(ขึ้น) = สัดส่วนขึ้นในวันที่ไม่ flat (ขึ้น vs ลง) แทนสัดส่วนขึ้นของทุกวัน
target="flat_dir" / "flat_price" (E9-FLAT): p_up = สัดส่วนวันนิ่งในอดีต
"""
import numpy as np

from forecast.models.common import as_frame, finite


class BaselineModel:
    def __init__(self, quantiles, target="up"):
        self.quantiles, self.target = quantiles, target

    def fit(self, X, up, z, meta=None):
        up, z = np.asarray(up, float), np.asarray(z, float)
        if self.target != "up":   # updown / flat_dir / flat_price — นิยามเดียวกับ E9 (models/e9.py)
            from forecast.models.e9 import target_rows
            u = target_rows(meta, self.target)[1].astype(float)
        else:
            u = up[finite(up)]
        self.p_up_ = float(u.mean()) if len(u) else 0.5
        zz = z[finite(z)]
        if len(zz):
            q = np.quantile(zz, self.quantiles)
            self.zq_ = q - np.median(zz)   # ค่ากลาง = ราคาล่าสุดที่รู้ (z = 0)
        else:
            self.zq_ = np.zeros(len(self.quantiles))
        return self

    def predict(self, X, meta=None):
        n = len(X)
        return as_frame(np.full(n, self.p_up_), np.tile(self.zq_, (n, 1)), self.quantiles, X.index)
