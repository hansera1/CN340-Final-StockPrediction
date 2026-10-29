"""แบบ A / E0: ราคา = ราคาล่าสุดที่รู้, ทิศทาง = ความถี่ "ขึ้น" ในอดีต"""
import numpy as np

from forecast.models.common import as_frame, finite


class BaselineModel:
    def __init__(self, quantiles):
        self.quantiles = quantiles

    def fit(self, X, up, z, meta=None):
        up, z = np.asarray(up, float), np.asarray(z, float)
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
