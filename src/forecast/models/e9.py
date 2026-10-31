"""E9: โมเดลตารางรอบที่ 2 — ทิศทาง "ขึ้น vs ลง" + ช่วงราคา conformal ปรับตามความผันผวน

- ทิศทางฝึกเฉพาะวันที่ไม่ flat (meta["direction"] != 0) เพราะวัน flat ไม่มีคำตอบที่ถูก
  (E9-FLAT ใช้คลาสเดียวกันกับเป้าหมายอื่น: target = flat_dir / flat_price / price_updown → target_rows)
  ผลลัพธ์อยู่ในคอลัมน์ p_up เสมอ = P(label = 1) ของเป้าหมายนั้น
- ราคาจุด = ราคาล่าสุด ณ จุดตัด (z = 0); ช่วงราคา = quantile ของ z / σ̂ จาก 60 วันล่าสุดของชุดฝึก × σ̂ ของวันที่ทาย
  σ̂ = sig20 (ค่าเฉลี่ย |z| ของรอบเดียวกัน 20 วันก่อนหน้า — features/dataset.py → add_sigma)
- การเลือก hyperparameter ใช้ time-series CV ภายในชุดฝึกเท่านั้น (ไม่แตะช่วงทดสอบ)
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.models.common import as_frame


def updown(meta):
    """แถวที่ไม่ flat และ label ขึ้น (1) / ลง (0)"""
    d = meta["direction"].to_numpy(float)
    m = np.isfinite(d) & (d != 0)
    return m, (d[m] == 1).astype(int)


def same_as_ref(meta):
    """ราคาปลายช่วง = P(c) (ราคาล่าสุด ณ จุดตัด) — เทียบด้วย tolerance กัน float error จาก CSV"""
    pe, pr = meta["p_end"].to_numpy(float), meta["p_ref"].to_numpy(float)
    ok = np.isfinite(pe) & np.isfinite(pr)
    return ok, np.isclose(pe, pr, rtol=0, atol=1e-6)


def target_rows(meta, target):
    """แถวที่ใช้ฝึก + label 0/1 ตามเป้าหมาย
    updown       ขึ้น vs ลง บนวันไม่ flat (E9)
    flat_dir     ราคาปลาย = ราคาต้นช่วง (E9-FLAT A-dir) — ทุกวัน
    flat_price   ราคาปลาย = P(c) (E9-FLAT A-price) — ทุกวัน
    price_updown ราคาปลาย > P(c) บนวันที่ราคาปลาย ≠ P(c) (E9-FLAT p_price)"""
    if target == "updown":
        return updown(meta)
    if target == "flat_dir":
        d = meta["direction"].to_numpy(float)
        m = np.isfinite(d)
        return m, (d[m] == 0).astype(int)
    ok, same = same_as_ref(meta)
    if target == "flat_price":
        return ok, same[ok].astype(int)
    if target == "price_updown":
        m = ok & ~same
        return m, (meta["p_end"].to_numpy(float)[m] > meta["p_ref"].to_numpy(float)[m]).astype(int)
    raise ValueError(f"ไม่รู้จัก target: {target}")


def inner_folds(dates, n_folds=3, val_frac=0.15):
    """time-series CV ภายในชุดฝึก: validation = n_folds บล็อกท้าย (บล็อกละ val_frac ของวัน), train = ทุกวันก่อนบล็อก"""
    ud = np.unique(dates)
    size = max(1, int(len(ud) * val_frac))
    for k in range(n_folds, 0, -1):
        lo, hi = len(ud) - k * size, len(ud) - (k - 1) * size
        if lo < size:
            continue
        yield np.isin(dates, ud[:lo]), np.isin(dates, ud[lo:hi])


def last_days_split(dates, val_frac=0.2):
    ud = np.unique(dates)
    cut = ud[int(len(ud) * (1 - val_frac))]
    return dates < cut, dates >= cut


class ConformalPrice:
    def __init__(self, quantiles, window_days=60):
        self.quantiles, self.window = quantiles, window_days

    def fit(self, meta):
        z, s = meta["z"].to_numpy(float), meta["sig20"].to_numpy(float)
        ok = np.isfinite(z) & np.isfinite(s) & (s > 0)
        dates = meta["date"].to_numpy()
        recent = np.isin(dates, np.unique(dates[ok])[-self.window:]) & ok
        self.uq_ = np.quantile(z[recent] / s[recent], self.quantiles)
        self.uq_[list(self.quantiles).index(0.5)] = 0.0          # ราคาจุด = ราคาล่าสุด
        self.sig_fill_ = float(np.median(s[ok]))
        return self

    def predict(self, meta):
        s = meta["sig20"].to_numpy(float)
        s = np.where(np.isfinite(s) & (s > 0), s, self.sig_fill_)
        return np.outer(s, self.uq_)


class _E9Base:
    needs_sequences = False

    def _X(self, X):
        if getattr(self, "interaction", False) and {"f_r_open_to_cut", "f_rv_intra"} <= set(X.columns):
            X = X.assign(f_ix_open_x_rv=X["f_r_open_to_cut"] * X["f_rv_intra"])
        return X

    def fit(self, X, up, z, meta):
        self.price_ = ConformalPrice(self.quantiles).fit(meta)
        m, y = target_rows(meta, getattr(self, "target", "updown"))
        X = self._X(X).loc[m]
        self.cols_ = X.columns[X.notna().any()]               # ตัดคอลัมน์ที่ว่างทั้งหมด
        self._fit_dir(X[self.cols_], y, meta["date"].to_numpy()[m])
        return self

    def predict(self, X, meta):
        X = self._X(X).reindex(columns=self.cols_)
        p, extra = self._predict_dir(X)
        out = as_frame(p, self.price_.predict(meta), self.quantiles, X.index)
        for k, v in extra.items():
            out[k] = v
        return out


class E9LR(_E9Base):
    """Logistic L2 — เลือก C ด้วย time-series CV ภายใน แล้ว bagging ด้วย block bootstrap ตามเวลา"""

    def __init__(self, quantiles, Cs=(0.003, 0.01, 0.03, 0.1), n_bags=20, block_days=20, interaction=False,
                 seed=0, target="updown"):
        self.quantiles, self.Cs, self.n_bags, self.block_days = quantiles, Cs, n_bags, block_days
        self.interaction, self.seed, self.target = interaction, seed, target

    @staticmethod
    def _pipe(C):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))

    def _fit_dir(self, X, y, dates):
        scores = {}
        for C in self.Cs:
            ll = [log_loss(y[va], self._pipe(C).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1], labels=[0, 1])
                  for tr, va in inner_folds(dates) if len(np.unique(y[tr])) == 2]
            scores[C] = np.mean(ll) if ll else np.inf
        self.C_ = min(scores, key=scores.get) if np.isfinite(min(scores.values())) else self.Cs[-1]
        ud = np.unique(dates)
        blocks = [ud[i:i + self.block_days] for i in range(0, len(ud), self.block_days)]
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for _ in range(self.n_bags):
            pick = np.concatenate([np.flatnonzero(np.isin(dates, blocks[j]))
                                   for j in rng.integers(0, len(blocks), len(blocks))])
            if len(np.unique(y[pick])) == 2:
                self.models_.append(self._pipe(self.C_).fit(X.iloc[pick], y[pick]))

    def _predict_dir(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return p, {"C": self.C_}


class E9GBM(_E9Base):
    """LightGBM ตื้น — เลือก min_data_in_leaf ด้วย time-series CV ภายใน, early stopping บน 20% วันท้าย
    แล้วฝึกใหม่บนชุดฝึกทั้งหมดด้วยจำนวนต้นที่ได้, เฉลี่ย 5 seed; ส่ง SHAP (TreeSHAP ของ LightGBM) ต่อแถว"""

    PARAMS = dict(objective="binary", max_depth=2, num_leaves=4, learning_rate=0.03, lambda_l2=10.0,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)

    def __init__(self, quantiles, min_leafs=(30, 50, 100), seeds=(0, 1, 2, 3, 4), max_trees=500, patience=50,
                 interaction=False, target="updown"):
        self.quantiles, self.min_leafs, self.seeds = quantiles, min_leafs, seeds
        self.max_trees, self.patience, self.interaction, self.target = max_trees, patience, interaction, target

    def _train(self, X, y, Xv, yv, mdl, seed):
        import lightgbm as lgb
        p = dict(self.PARAMS, min_data_in_leaf=mdl, seed=seed)
        dtr = lgb.Dataset(X, y, free_raw_data=False)
        dva = lgb.Dataset(Xv, yv, reference=dtr)
        b = lgb.train(p, dtr, self.max_trees, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        return b, b.best_score["valid_0"]["binary_logloss"], max(1, b.best_iteration)

    def _fit_dir(self, X, y, dates):
        import lightgbm as lgb
        scores = {}
        for mdl in self.min_leafs:
            s = [self._train(X[tr], y[tr], X[va], y[va], mdl, 0)[1] for tr, va in inner_folds(dates)]
            scores[mdl] = np.mean(s) if s else np.inf
        self.mdl_ = min(scores, key=scores.get)
        tr, va = last_days_split(dates)
        self.boosters_ = []
        for seed in self.seeds:
            _, _, n = self._train(X[tr], y[tr], X[va], y[va], self.mdl_, seed)
            p = dict(self.PARAMS, min_data_in_leaf=self.mdl_, seed=seed)
            self.boosters_.append(lgb.train(p, lgb.Dataset(X, y), n))

    def _predict_dir(self, X):
        p = np.mean([b.predict(X) for b in self.boosters_], axis=0)
        shap = np.mean([b.predict(X, pred_contrib=True)[:, :-1] for b in self.boosters_], axis=0)
        extra = {f"shap_{c}": shap[:, i] for i, c in enumerate(self.cols_)}
        extra["min_data_in_leaf"] = self.mdl_
        extra["n_trees"] = np.mean([b.num_trees() for b in self.boosters_])
        return p, extra


class E9Zoo(_E9Base):
    """ภาคผนวก (model zoo): ค่า default ที่กำหนดล่วงหน้า ไม่จูน — feature / walk-forward / ราคาเดียวกับ E9
    ทุกตัวผ่าน median impute + standardize; SVM ใช้ Platt calibration ด้วย time-series CV 3 ช่วง"""

    def __init__(self, quantiles, kind="xgb", seed=0):
        self.quantiles, self.kind, self.seed = quantiles, kind, seed

    def _estimator(self):
        if self.kind == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(random_state=self.seed, n_jobs=4, verbosity=0)
        if self.kind == "cat":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(random_seed=self.seed, thread_count=4, verbose=0, allow_writing_files=False)
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=self.seed, n_jobs=4)
        if self.kind in ("svm_linear", "svm_rbf"):
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.svm import SVC
            return CalibratedClassifierCV(SVC(kernel=self.kind[4:]), method="sigmoid", cv=TimeSeriesSplit(3))
        raise ValueError(f"ไม่รู้จัก kind: {self.kind}")

    def _fit_dir(self, X, y, dates):
        order = np.argsort(dates, kind="stable")          # TimeSeriesSplit ต้องเรียงตามเวลา
        self.model_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                    self._estimator()).fit(X.iloc[order], y[order])

    def _predict_dir(self, X):
        return self.model_.predict_proba(X)[:, 1], {}
