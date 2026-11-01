"""E13 ชั้น 2: offset logistic (สเปกล็อกใน results/e13_preregistration.md)

logit(p_final) = logit(p₁) + xᵀβ · logit(p₁) เป็น offset (สัมประสิทธิ์ 1) · ไม่มี intercept
x standardize ด้วย mean / sd ของชุดฝึก · ลงโทษ (λ/2)·‖β‖² บน log-likelihood รวม (λ = 45 ตายตัว) · L-BFGS
"""
import numpy as np
from scipy.optimize import minimize

from forecast.features.fine import F5

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Stage2:
    def __init__(self, l2=45.0):
        self.l2 = float(l2)

    def fit(self, X, y, p1):
        X, y, off = np.asarray(X, float), np.asarray(y, float), logit(p1)
        self.mean_ = X.mean(0)
        sd = X.std(0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        Z = (X - self.mean_) / self.sd_

        def loss(b):
            eta = off + Z @ b
            nll = np.sum(np.logaddexp(0, eta) - y * eta)
            grad = Z.T @ (1 / (1 + np.exp(-eta)) - y) + self.l2 * b
            return nll + 0.5 * self.l2 * b @ b, grad

        res = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
        self.beta_ = res.x
        return self

    def predict(self, X, p1):
        Z = (np.asarray(X, float) - self.mean_) / self.sd_
        return 1 / (1 + np.exp(-(logit(p1) + Z @ self.beta_)))

    def to_dict(self):
        return {"l2": self.l2, "mean": self.mean_.tolist(), "sd": self.sd_.tolist(), "beta": self.beta_.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["l2"])
        m.mean_, m.sd_, m.beta_ = (np.asarray(d[k], float) for k in ["mean", "sd", "beta"])
        return m


def training_rows(table, date, train_days=20):
    """table: [date, symbol, p1, direction, F5...] → (แถวฝึกไม่ flat, วันที่ใช้)
    วันฝึก = train_days วันล่าสุดก่อน date ที่มีแถวใช้ได้ (มี label, f5 ครบ, p₁ out-of-sample)"""
    ok = table.dropna(subset=["p1", "direction", *F5])
    days = sorted(ok.loc[ok["date"] < date, "date"].unique())[-train_days:]
    return ok[ok["date"].isin(days) & (ok["direction"] != 0)], days


def fit_for_day(table, date, train_days=20, l2=45.0, min_nonflat=20):
    """→ (Stage2 หรือ None, info) · แถวไม่ flat < min_nonflat → None (p_final = p₁)"""
    tr, days = training_rows(table, date, train_days)
    info = {"date": str(date), "n_days": len(days), "n_nonflat": len(tr),
            "first_day": str(days[0]) if days else None, "last_day": str(days[-1]) if days else None}
    if len(days) < train_days or len(tr) < min_nonflat:
        info["reason"] = f"วันฝึก {len(days)}/{train_days}, แถวไม่ flat {len(tr)} (< {min_nonflat}) → p_final = p₁"
        return None, info
    m = Stage2(l2).fit(tr[F5].to_numpy(float), (tr["direction"] == 1).to_numpy(float), tr["p1"].to_numpy(float))
    info["reason"] = None
    return m, info
