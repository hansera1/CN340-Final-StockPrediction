"""แบบ B / E1: ทิศทาง (Logistic) และราคา (quantile GBM) แยกกัน"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.models.common import as_frame, finite


def direction_model(C=0.3):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=C, max_iter=2000))


class DirectionModel:
    """Logistic แบบ regularized; ถ้าข้อมูลมีคลาสเดียวใช้ base rate"""

    def __init__(self, C=0.3):
        self.C = C

    def fit(self, X, up):
        up = np.asarray(up, float)
        m = finite(up)
        self.const_ = None
        if len(np.unique(up[m])) < 2:
            self.const_ = float(up[m].mean()) if m.any() else 0.5
        else:
            self.model_ = direction_model(self.C).fit(X[m], up[m].astype(int))
        return self

    def predict_proba(self, X):
        if self.const_ is not None:
            return np.full(len(X), self.const_)
        return self.model_.predict_proba(X)[:, 1]


class QuantileBank:
    """GBM ตื้น ต่อ quantile — จัดการค่าหาย (NaN) ได้เอง"""

    def __init__(self, quantiles, **params):
        self.quantiles = quantiles
        self.params = dict(max_depth=3, learning_rate=0.05, max_iter=200,
                           min_samples_leaf=20, random_state=0) | params

    def fit(self, X, z):
        z = np.asarray(z, float)
        m = finite(z)
        # ตัดคอลัมน์ที่ว่างทั้งหมดในชุดฝึก (เช่น features ภายในวันของรอบเช้า)
        self.cols_ = X.columns[X[m].notna().any()]
        self.models_ = [HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params)
                        .fit(X.loc[m, self.cols_], z[m]) for q in self.quantiles]
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X[self.cols_]) for m in self.models_])


class IndependentModel:
    def __init__(self, quantiles, C=0.3):
        self.quantiles = quantiles
        self.direction = DirectionModel(C)
        self.price = QuantileBank(quantiles)

    def fit(self, X, up, z, meta=None):
        self.direction.fit(X, up)
        self.price.fit(X, z)
        return self

    def predict(self, X, meta=None):
        return as_frame(self.direction.predict_proba(X), self.price.predict(X), self.quantiles, X.index)
