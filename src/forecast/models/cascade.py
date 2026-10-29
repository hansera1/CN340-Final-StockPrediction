"""แบบ C / E2: โมเดลสองขั้น

ขั้น 1: ข้อมูลที่รู้ก่อนส่ง → probability ทิศทาง
ขั้น 2: features เดิม + probability แบบ temporal OOF → quantiles ของ z

temporal OOF: แบ่งวันในชุดฝึกเป็นบล็อกตามเวลา บล็อก i ได้ probability จากโมเดลที่ฝึกบนบล็อก < i เท่านั้น
บล็อกแรกสร้างไม่ได้ จึงถูกตัดออกจากการฝึกขั้นสอง (ห้ามใช้ OOF แบบสุ่ม)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from forecast.models.common import as_frame, finite
from forecast.models.independent import DirectionModel, QuantileBank


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


class CascadeModel:
    def __init__(self, quantiles, n_oof_blocks=5, calibrate=False, C=0.3):
        self.quantiles = quantiles
        self.n_oof_blocks = n_oof_blocks
        self.calibrate = calibrate
        self.C = C

    def _temporal_oof(self, X, up, dates):
        oof = np.full(len(X), np.nan)
        blocks = np.array_split(np.unique(dates), self.n_oof_blocks)
        for i in range(1, len(blocks)):
            tr = np.isin(dates, np.concatenate(blocks[:i]))
            te = np.isin(dates, blocks[i])
            oof[te] = DirectionModel(self.C).fit(X[tr], up[tr]).predict_proba(X[te])
        return oof

    def _stage2_X(self, X, p):
        X2 = X.copy()
        X2["p_up_stage1"] = p
        return X2

    def fit(self, X, up, z, meta):
        up = np.asarray(up, float)
        dates = meta["date"].to_numpy()
        oof = self._temporal_oof(X, up, dates)

        self.calibrator_ = None
        m = finite(oof, up)
        if self.calibrate and len(np.unique(up[m])) == 2:
            # Platt จาก OOF (อดีตเท่านั้น) — ตรวจผลด้วย log loss / Brier เสมอ
            self.calibrator_ = LogisticRegression().fit(_logit(oof[m]), up[m].astype(int))
            oof[m] = self.calibrator_.predict_proba(_logit(oof[m]))[:, 1]

        self.stage1_ = DirectionModel(self.C).fit(X, up)
        keep = finite(oof)
        self.stage2_ = QuantileBank(self.quantiles).fit(self._stage2_X(X[keep], oof[keep]),
                                                        np.asarray(z, float)[keep])
        return self

    def predict(self, X, meta=None):
        p = self.stage1_.predict_proba(X)
        if self.calibrator_ is not None:
            p = self.calibrator_.predict_proba(_logit(p))[:, 1]
        return as_frame(p, self.stage2_.predict(self._stage2_X(X, p)), self.quantiles, X.index)
