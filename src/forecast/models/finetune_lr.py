"""รอบ 7 E14 / E17a: pretrain แล้ว fine-tune โมเดล logistic — กรอบเดียวกันทั้งทิศทางและราคา

pretrain   LR L2 บนข้อมูลรวมหลายกลุ่ม → β₀, intercept₀ · C เลือกด้วย inner CV · fit ครั้งเดียวต่อรอบ fit
fine-tune  min Σ log loss + λ‖β − β₀‖² · λ = κ / (2·C_pretrain) · κ = ∞ คือใช้ β₀ ตรง ๆ
           scaler ตัวเดียวกับตอน pretrain · intercept อิสระ · dummy ที่คงที่ในชุด fine-tune ตัดออกและรวมเข้า intercept เริ่มต้น
           feature ที่ไม่อยู่ใน pretrain เริ่มที่ β₀ = 0 · bag เฉพาะ fine-tune 20 รอบ (บล็อก 20 วัน)
เลือก κ    inner CV แบบขยาย 3 fold (fold ละ ≥ 10 วัน) · pretrain ซ้ำในแต่ละ fold ด้วย C เดิม ไม่ให้ β₀ เห็นวัน validation
           ข้อมูล fine-tune < 40 วัน → κ = ∞ · < 60 วัน → κ คงที่ FIXED_KAPPA
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

CS = (0.003, 0.01, 0.03, 0.1)
KAPPAS = (0.3, 1.0, 3.0, 10.0, 30.0, np.inf)
FIXED_KAPPA = 10.0
MIN_DAYS, MIN_CV_DAYS = 40, 60


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class Scaler:
    """median impute + standardize จากแถว pretrain · คอลัมน์ที่ไม่มีค่าในแถวนั้น (NaN) → 0 หลัง standardize"""

    def __init__(self, X):
        self.median = np.nan_to_num(np.nanmedian(X, axis=0))
        filled = np.where(np.isnan(X), self.median, X)
        self.mean, self.std = filled.mean(0), filled.std(0)
        self.std[self.std == 0] = 1.0

    def __call__(self, X):
        return (np.where(np.isnan(X), self.median, X) - self.mean) / self.std


def inner_blocks(dates, n_folds=3, min_val=10, frac=0.15):
    """fold แบบขยาย → [(วันฝึก, วัน validation)] · validation = บล็อกท้าย n_folds บล็อก"""
    ud = np.unique(dates)
    size = max(min_val, int(len(ud) * frac))
    out = []
    for k in range(n_folds, 0, -1):
        start = len(ud) - k * size
        if start >= 2 * min_val:
            out.append((ud[:start], ud[start:start + size]))
    return out


def fit_pretrain(X, y, C):
    m = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    return m.coef_[0].copy(), float(m.intercept_[0])


def choose_c(X, y, dates):
    scores = {}
    for C in CS:
        ll = []
        for tr, va in inner_blocks(dates):
            a, b = np.isin(dates, tr), np.isin(dates, va)
            if len(np.unique(y[a])) == 2:
                ll.append(logloss(y[b], LogisticRegression(C=C, max_iter=2000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]).mean())
        scores[C] = np.mean(ll) if ll else np.inf
    return min(scores, key=scores.get)


def fit_finetune(X, y, beta0, a0, lam):
    """→ (β, intercept) · lam = np.inf → β₀ ตรง ๆ"""
    if not np.isfinite(lam) or len(np.unique(y)) < 2:
        return beta0.copy(), a0

    def f(w):
        b, a = w[:-1], w[-1]
        s = X @ (beta0 + b) + a
        p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        loss = np.logaddexp(0, -s)[y == 1].sum() + np.logaddexp(0, s)[y == 0].sum() + lam * b @ b
        g = (p - y)
        return loss, np.r_[X.T @ g + 2 * lam * b, g.sum()]

    w = minimize(f, np.r_[np.zeros(X.shape[1]), a0], jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return beta0 + w[:-1], float(w[-1])


class PretrainFineTune:
    """pre_cols: คอลัมน์ pretrain (รวม dummy) · ft_cols: คอลัมน์ตอน fine-tune · const: {dummy: ค่าคงที่ในกลุ่มนี้} → รวมเข้า intercept"""

    def __init__(self, ft_days_min=MIN_DAYS, n_bags=20, block_days=20, seed=0):
        self.ft_days_min, self.n_bags, self.block_days, self.seed = ft_days_min, n_bags, block_days, seed

    def _start(self, beta_pre, a_pre, pre_cols, ft_cols, const):
        beta0 = np.array([beta_pre[pre_cols.index(c)] if c in pre_cols else 0.0 for c in ft_cols])
        a0 = a_pre + sum(beta_pre[pre_cols.index(c)] * self.std_const(c, v, pre_cols) for c, v in const.items())
        return beta0, a0

    def std_const(self, c, v, pre_cols):
        i = pre_cols.index(c)
        return (v - self.scaler.mean[i]) / self.scaler.std[i]

    def fit(self, pre, ft, pre_cols, ft_cols, const, pm_only=None, inner_val=None):
        """pre / ft: dict(X = DataFrame, y, dates) · ft ต้องอยู่ในช่วงเดียวกับ pre · inner_val: dict(X, y, dates) แถว validation ตอนเลือก κ
        (ไม่ระบุ = แถว fine-tune เอง) → self"""
        Xp, yp, dp = pre["X"][pre_cols].to_numpy(float), pre["y"], pre["dates"]
        self.scaler = Scaler(Xp)
        Zp = self.scaler(Xp)
        self.C = choose_c(Zp, yp, dp)
        beta_pre, a_pre = fit_pretrain(Zp, yp, self.C)
        # คอลัมน์ที่ไม่อยู่ใน pretrain: scaler ของตัวเองจากแถวที่มีค่า (ชุด fine-tune ก่อนวันทดสอบ)
        self.pre_cols, self.ft_cols, self.const = pre_cols, ft_cols, const
        extra = [c for c in ft_cols if c not in pre_cols]
        self.extra_scaler = Scaler(ft["X"][extra].to_numpy(float)) if extra else None
        self.extra = extra
        Zf = self._z(ft["X"])
        yf, df = ft["y"], ft["dates"]
        beta0, a0 = self._start(beta_pre, a_pre, pre_cols, ft_cols, const)
        n_days = len(np.unique(df))
        self.inner_ll = np.nan
        if n_days < self.ft_days_min:
            kappa = np.inf
        elif n_days < MIN_CV_DAYS:
            kappa = FIXED_KAPPA
        else:
            kappa, self.inner_ll = self._choose_kappa(pre, ft, inner_val)
        self.kappa = kappa
        lam = kappa / (2 * self.C)
        self.models = []
        if not np.isfinite(kappa):
            self.models = [(beta0, a0)]
        else:
            ud = np.unique(df)
            blocks = [ud[i:i + self.block_days] for i in range(0, len(ud), self.block_days)]
            rng = np.random.default_rng(self.seed)
            for _ in range(self.n_bags):
                pick = np.concatenate([np.flatnonzero(np.isin(df, blocks[j])) for j in rng.integers(0, len(blocks), len(blocks))])
                if len(np.unique(yf[pick])) == 2:
                    self.models.append(fit_finetune(Zf[pick], yf[pick], beta0, a0, lam))
            if not self.models:
                self.models = [(beta0, a0)]
        return self

    def _z(self, X):
        pre = [c for c in self.ft_cols if c in self.pre_cols]
        Z = np.zeros((len(X), len(self.ft_cols)))
        Zp = self.scaler(X.reindex(columns=self.pre_cols).to_numpy(float))
        for j, c in enumerate(self.ft_cols):
            if c in pre:
                Z[:, j] = Zp[:, self.pre_cols.index(c)]
        if self.extra:
            Ze = self.extra_scaler(X[self.extra].to_numpy(float))
            for k, c in enumerate(self.extra):
                Z[:, self.ft_cols.index(c)] = Ze[:, k]
        return Z

    def _choose_kappa(self, pre, ft, inner_val):
        val = inner_val or ft
        folds = inner_blocks(val["dates"])
        scores = {k: [] for k in KAPPAS}
        for tr, va in folds:
            cut = np.min(va)
            a = pre["dates"] < cut
            s = Scaler(pre["X"][self.pre_cols].to_numpy(float)[a])
            Zp = s(pre["X"][self.pre_cols].to_numpy(float)[a])
            if len(np.unique(pre["y"][a])) < 2:
                continue
            beta_pre, a_pre = fit_pretrain(Zp, pre["y"][a], self.C)
            keep, self.scaler = self.scaler, s
            beta0, a0 = self._start(beta_pre, a_pre, self.pre_cols, self.ft_cols, self.const)
            f = ft["dates"] < cut
            Zf = self._z(ft["X"].iloc[np.flatnonzero(f)])
            m = np.isin(val["dates"], va)
            Zv = self._z(val["X"].iloc[np.flatnonzero(m)])
            self.scaler = keep
            for k in KAPPAS:
                days_ok = len(np.unique(ft["dates"][f])) >= self.ft_days_min
                b, i = fit_finetune(Zf, ft["y"][f], beta0, a0, k / (2 * self.C)) if (days_ok and np.isfinite(k)) else (beta0, a0)
                scores[k].append(logloss(val["y"][m], 1 / (1 + np.exp(-(Zv @ b + i)))).mean())
        mean = {k: np.mean(v) for k, v in scores.items() if v}
        if not mean:
            return np.inf, np.nan
        best = min(mean, key=mean.get)
        return best, mean[best]

    def predict(self, X):
        Z = self._z(X)
        return np.mean([1 / (1 + np.exp(-(Z @ b + a))) for b, a in self.models], axis=0)
