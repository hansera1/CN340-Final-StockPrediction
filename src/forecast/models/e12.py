"""E12: โมเดลจากงานรายวันเดิม (CN340) ค่าตายตัว ไม่จูน — เป้าหมาย / walk-forward / ราคาเดียวกับ E9

RF / XGB / MLP ผ่าน median impute (+ standardize สำหรับ MLP) แล้วเฉลี่ยความน่าจะเป็นหลาย seed
(LR ใช้ E9LR ตามสเปก E9-LR)
ทุกโมเดล (รวม E9LR ผ่าน permutation=True) ส่งคอลัมน์ perm_<feature> = P(ขึ้น) เมื่อแทนค่า feature นั้นด้วยค่าสุ่มจากชุดฝึก
→ permutation importance นอกตัวอย่าง (Δ log loss) คิดใน experiments/12_score_e12.py
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.models.e9 import E9LR, _E9Base

RF = dict(n_estimators=400, max_depth=4, min_samples_leaf=50, max_features=0.5, n_jobs=1)
XGB = dict(n_estimators=300, max_depth=2, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           n_jobs=1, verbosity=0)
MLP = dict(hidden_layer_sizes=(8, 4), activation="relu", alpha=7.0, solver="adam", learning_rate_init=1e-3,
           max_iter=3000)
SEEDS = {"rf": 5, "xgb": 5, "mlp": 10}


class _Permute:
    """เก็บตัวอย่างชุดฝึกไว้ แล้วตอนทำนายคืน P(ขึ้น) เมื่อสุ่มค่าแต่ละ feature ใหม่จากชุดฝึก"""

    def _keep_train(self, X):
        self.X_train_ = X

    def _perm_extra(self, X, predict):
        if not getattr(self, "permutation", False):
            return {}
        rng = np.random.default_rng(12)
        out = {}
        for c in X.columns:
            Xp = X.copy()
            Xp[c] = rng.choice(self.X_train_[c].to_numpy(), len(X))
            out[f"perm_{c}"] = predict(Xp)
        return out


class E12Fixed(_Permute, _E9Base):
    def __init__(self, quantiles, kind="rf", n_seeds=None, permutation=True, target="updown"):
        self.quantiles, self.kind, self.permutation, self.target = quantiles, kind, permutation, target
        self.n_seeds = n_seeds or SEEDS[kind]

    def _estimator(self, seed):
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                 RandomForestClassifier(random_state=seed, **RF))
        if self.kind == "xgb":
            from xgboost import XGBClassifier
            return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                 XGBClassifier(random_state=seed, **XGB))
        if self.kind == "mlp":
            from sklearn.neural_network import MLPClassifier
            return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                 MLPClassifier(random_state=seed, **MLP))
        raise ValueError(f"ไม่รู้จัก kind: {self.kind}")

    def _fit_dir(self, X, y, dates):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        self._keep_train(X)
        self.models_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for s in range(self.n_seeds):
                self.models_.append(self._estimator(s).fit(X, y))
        if self.kind == "mlp":   # converge = หยุดก่อนครบ max_iter (tol ของ sklearn)
            self.converged_ = float(np.mean([m[-1].n_iter_ < MLP["max_iter"] for m in self.models_]))

    def _proba(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)

    def _predict_dir(self, X):
        extra = self._perm_extra(X, self._proba)
        if self.kind == "mlp":
            extra["mlp_converged"] = self.converged_
        return self._proba(X), extra


class E12LR(_Permute, E9LR):
    """สเปก E9-LR (เลือก C ด้วย CV ภายใน + bagging) + permutation importance"""

    def __init__(self, quantiles, permutation=True, **kw):
        super().__init__(quantiles, **kw)
        self.permutation = permutation

    def _fit_dir(self, X, y, dates):
        self._keep_train(X)
        super()._fit_dir(X, y, dates)

    def _predict_dir(self, X):
        p, extra = super()._predict_dir(X)
        f = lambda Z: np.mean([m.predict_proba(Z)[:, 1] for m in self.models_], axis=0)
        return p, {**extra, **self._perm_extra(X, f)}
