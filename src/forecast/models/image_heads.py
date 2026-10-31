"""E7: หัวทำนายบน feature 512 มิติจาก ResNet18 ที่ freeze (รับ embedding ผ่านช่อง seqs แบบเดียวกับ cnn1d)

ขั้น 0  FrozenLRModel:   standardize + logistic regression แบบ regularized
ขั้น 1  FrozenHeadModel: หัวใหม่ dropout + linear, AdamW, warmup สั้นแล้ว cosine decay,
                         early stopping บน validation ที่อยู่หลังช่วง train (log loss), หลาย seed เฉลี่ย
                         และเก็บ p ราย seed (p_up_seed<i>) ไว้รายงานค่าเฉลี่ย/ช่วง — ไม่เลือก seed ที่ดีที่สุด
ช่วงราคาใช้ baseline เชิงประจักษ์ (เหมือน cnn1d)
"""
import math

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from forecast.models.baseline import BaselineModel
from forecast.models.cnn1d import train_device
from forecast.models.common import as_frame, finite


class FrozenLRModel:
    needs_sequences = True

    def __init__(self, quantiles, C=0.01):
        self.quantiles, self.C = quantiles, C

    def fit(self, X, up, z, meta, emb):
        self.base_ = BaselineModel(self.quantiles).fit(X, up, z)
        up = np.asarray(up, float)
        m = finite(up)
        self.const_ = float(up[m].mean()) if len(np.unique(up[m])) < 2 else None
        if self.const_ is None:
            self.lr_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000)).fit(
                emb[m], up[m].astype(int))
        return self

    def predict(self, X, meta, emb):
        p = np.full(len(X), self.const_) if self.const_ is not None else self.lr_.predict_proba(emb)[:, 1]
        zq = self.base_.predict(X).drop(columns="p_up").to_numpy()
        return as_frame(p, zq, self.quantiles, X.index)


class FrozenHeadModel:
    needs_sequences = True

    def __init__(self, quantiles, seeds=(0, 1, 2, 3, 4), epochs=100, patience=8, lr=1e-3,
                 weight_decay=1e-3, dropout=0.5, warmup=5, val_frac=0.2, batch_size=64):
        self.quantiles, self.seeds = quantiles, seeds
        self.epochs, self.patience, self.lr, self.wd = epochs, patience, lr, weight_decay
        self.dropout, self.warmup, self.val_frac, self.bs = dropout, warmup, val_frac, batch_size

    def _train_one(self, seed, xtr, ytr, xva, yva):
        torch.manual_seed(seed)
        dev = train_device()
        head = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(xtr.shape[1], 1)).to(dev)
        xtr, ytr, xva, yva = (t.to(dev) for t in (xtr, ytr, xva, yva))
        opt = torch.optim.AdamW(head.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / self.warmup if e < self.warmup else
                                                  0.5 * (1 + math.cos(math.pi * (e - self.warmup) /
                                                                      max(1, self.epochs - self.warmup))))
        loss_fn = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(seed)
        best, state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            head.train()
            for idx in torch.randperm(len(xtr), generator=g).split(self.bs):
                opt.zero_grad()
                loss_fn(head(xtr[idx]).squeeze(-1), ytr[idx]).backward()
                opt.step()
            sched.step()
            head.eval()
            with torch.no_grad():
                v = loss_fn(head(xva).squeeze(-1), yva).item()
            if v < best - 1e-4:
                best, state, bad = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        head.load_state_dict(state)
        return head.cpu().eval()

    def fit(self, X, up, z, meta, emb):
        self.base_ = BaselineModel(self.quantiles).fit(X, up, z)
        up = np.asarray(up, float)
        m = finite(up)
        self.scaler_ = StandardScaler().fit(emb[m])
        x = torch.tensor(self.scaler_.transform(emb[m]), dtype=torch.float32)
        y = torch.tensor(up[m], dtype=torch.float32)
        ud = np.unique(meta["date"].to_numpy()[m])
        va = np.isin(meta["date"].to_numpy()[m], ud[int(len(ud) * (1 - self.val_frac)):])  # ช่วงท้ายตามเวลา
        self.heads_ = [self._train_one(s, x[~va], y[~va], x[va], y[va]) for s in self.seeds]
        return self

    def predict(self, X, meta, emb):
        x = torch.tensor(self.scaler_.transform(emb), dtype=torch.float32)
        with torch.no_grad():
            ps = np.stack([torch.sigmoid(h(x).squeeze(-1)).numpy() for h in self.heads_])
        zq = self.base_.predict(X).drop(columns="p_up").to_numpy()
        out = as_frame(ps.mean(axis=0), zq, self.quantiles, X.index)
        for i, p in enumerate(ps):
            out[f"p_up_seed{i}"] = np.clip(p, 1e-4, 1 - 1e-4)
        return out
