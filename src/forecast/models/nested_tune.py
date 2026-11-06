"""รอบ 7 E15 / E17b: nested walk-forward — หลายโมเดล + feature selection + Optuna

fit ใหม่ทุกบล็อกทดสอบ (5 วัน) · จูนใหม่ทุก RETUNE_DAYS วันทำการด้วยข้อมูลก่อนวันทดสอบเท่านั้น
inner CV 4 fold แบบขยาย gap 5 วัน วัดด้วย log loss · ลำดับ feature = permutation importance บนส่วนท้ายของ inner-train
(แยกจาก validation) ด้วยค่าตั้งต้น ครั้งเดียวต่อ fold ต่อโมเดล · Optuna เลือก k · ElasticNet ให้ L1 เลือกเอง
"""
import warnings

import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*penalty.*")

MODELS = ["lr", "enet", "lgbm", "xgb", "rf", "svm", "mlp"]
SKLEARN_ONLY = {"lr", "enet", "rf", "svm", "mlp"}
KS = (10, 15, 20, 30, None)
RETUNE_DAYS, N_TRIALS, GAP = 120, 25, 5


def _prep():
    return [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]


def build(name, p, seed=0):
    if name == "lr":
        return make_pipeline(*_prep(), LogisticRegression(C=p.get("C", 0.03), max_iter=2000))
    if name == "enet":
        return make_pipeline(*_prep(), LogisticRegression(C=p.get("C", 0.03), l1_ratio=p.get("l1_ratio", 0.5), solver="saga", max_iter=2000, random_state=seed))
    if name == "svm":
        return make_pipeline(*_prep(), CalibratedClassifierCV(LinearSVC(C=p.get("C", 0.03), max_iter=5000, random_state=seed), cv=3))
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=p.get("n", 200), max_depth=p.get("depth", 3), min_samples_leaf=p.get("leaf", 30),
                                                    max_features=p.get("mf", 0.5), random_state=seed, n_jobs=1))
    if name == "mlp":
        return make_pipeline(*_prep(), MLPClassifier(hidden_layer_sizes=p.get("hidden", (8,)), alpha=p.get("alpha", 3.0), max_iter=300, random_state=seed))
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=p.get("n", 150), learning_rate=p.get("lr", 0.03), max_depth=p.get("depth", 3), num_leaves=p.get("leaves", 7),
                                  min_child_samples=p.get("leaf", 30), subsample=p.get("sub", 0.8), subsample_freq=1, colsample_bytree=p.get("col", 0.8),
                                  reg_lambda=p.get("l2", 3.0), random_state=seed, n_jobs=1, verbose=-1)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=p.get("n", 150), learning_rate=p.get("lr", 0.03), max_depth=p.get("depth", 3), min_child_weight=p.get("leaf", 30) * 0.25,
                                 subsample=p.get("sub", 0.8), colsample_bytree=p.get("col", 0.8), reg_lambda=p.get("l2", 3.0), tree_method="hist", device="cpu",
                                 n_jobs=1, random_state=seed, verbosity=0)
    raise ValueError(name)


def suggest(trial, name):
    if name in ("lr", "svm"):
        return {"C": trial.suggest_float("C", 0.001, 1.0, log=True)}
    if name == "enet":
        return {"C": trial.suggest_float("C", 0.001, 1.0, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95)}
    if name == "rf":
        return {"n": trial.suggest_int("n", 100, 300, step=50), "depth": trial.suggest_int("depth", 2, 4), "leaf": trial.suggest_int("leaf", 30, 120),
                "mf": trial.suggest_float("mf", 0.2, 0.8)}
    if name == "mlp":
        layers = trial.suggest_int("layers", 1, 2)
        hidden = tuple(trial.suggest_int(f"u{i}", 4, 16) for i in range(layers))
        return {"hidden": hidden, "alpha": trial.suggest_float("alpha", 1.0, 100.0, log=True)}
    p = {"n": trial.suggest_int("n", 50, 300, step=50), "lr": trial.suggest_float("lr", 0.005, 0.05, log=True), "depth": trial.suggest_int("depth", 2, 4),
         "leaf": trial.suggest_int("leaf", 30, 120), "sub": trial.suggest_float("sub", 0.5, 1.0), "col": trial.suggest_float("col", 0.3, 1.0),
         "l2": trial.suggest_float("l2", 0.1, 30.0, log=True)}
    if name == "lgbm":
        p["leaves"] = trial.suggest_int("leaves", 3, 15)
    return p


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def proba(m, X):
    return m.predict_proba(X)[:, 1]


def tune_folds(dates, n=4, gap=GAP, val_frac=0.12, min_val=10):
    """→ [(วัน inner-train, วัน validation)] ขยายตามเวลา เว้น gap วันก่อน validation"""
    ud = np.unique(dates)
    size = max(min_val, int(len(ud) * val_frac))
    out = []
    for k in range(n, 0, -1):
        start = len(ud) - k * size
        if start - gap >= 40:
            out.append((ud[:start - gap], ud[start:start + size]))
    return out


def rank_features(name, X, y, dates, seed=0, tail=0.2):
    """permutation importance (log loss) บนส่วนท้ายของ inner-train ด้วยค่าตั้งต้น → index เรียงจากสำคัญมากไปน้อย"""
    ud = np.unique(dates)
    cut = ud[int(len(ud) * (1 - tail))]
    a, b = dates < cut, dates >= cut
    if len(np.unique(y[a])) < 2 or b.sum() == 0:
        return np.arange(X.shape[1])
    m = build(name, {}, seed).fit(X[a], y[a])
    base = logloss(y[b], proba(m, X[b])).mean()
    rng = np.random.default_rng(seed)
    gain = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Z = X[b].copy()
        Z[:, j] = rng.permutation(Z[:, j])
        gain[j] = logloss(y[b], proba(m, Z)).mean() - base
    return np.argsort(-gain)


def tune(name, X, y, dates, group, seed=0, n_trials=N_TRIALS):
    """→ dict(params, k, inner_ll, e0_ll, n_trials, oof) · oof = {index แถว: p} ของ best trial · group = รหัสกลุ่มของ E0 (หุ้น)"""
    folds = []
    for tr, va in tune_folds(dates):
        a, b = np.isin(dates, tr), np.isin(dates, va)
        if len(np.unique(y[a])) == 2 and b.any():
            order = rank_features(name, X[a], y[a], dates[a], seed) if name != "enet" else np.arange(X.shape[1])
            folds.append((np.flatnonzero(a), np.flatnonzero(b), order))
    if not folds:
        return None
    e0 = np.mean([logloss(y[b], np.array([y[a][group[a] == g].mean() for g in group[b]])).mean() for a, b, _ in folds])
    oofs = {}

    def objective(trial):
        p = suggest(trial, name)
        k = None if name == "enet" else trial.suggest_categorical("k", [str(x) for x in KS])
        k = None if k in (None, "None") else int(k)
        scores, oof = [], {}
        for step, (a, b, order) in enumerate(folds):
            cols = order if k is None else order[:k]
            m = build(name, p, seed).fit(X[a][:, cols], y[a])
            q = proba(m, X[b][:, cols])
            scores.append(logloss(y[b], q).mean())
            oof.update(zip(b.tolist(), q.tolist()))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        oofs[trial.number] = oof
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    params = suggest(optuna.trial.FixedTrial(best.params), name)
    k = best.params.get("k")
    return {"params": params, "k": None if k in (None, "None") else int(k), "inner_ll": best.value, "e0_ll": float(e0),
            "n_trials": len(study.trials), "oof": oofs[best.number]}
