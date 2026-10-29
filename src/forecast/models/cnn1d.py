"""ส่วนเพิ่ม D / E4: 1D-CNN ขนาดเล็กบนลำดับแท่งก่อนจุดตัด (C1 รอบบ่าย, C2 รอบเช้า)

- conv 2 ชั้น + dropout + weight decay + early stopping บน validation ที่อยู่หลังช่วง train
- ฝึกหลาย seed ตามจำนวนที่กำหนดก่อน แล้วเฉลี่ย (ไม่เลือก seed ที่ดีที่สุด)
- ให้เฉพาะ probability ทิศทาง; ช่วงราคาใช้ baseline เชิงประจักษ์
"""
import numpy as np
import torch
from torch import nn

from forecast.features.sequences import N_CHANNELS
from forecast.models.baseline import BaselineModel
from forecast.models.common import as_frame, finite


class _Net(nn.Module):
    def __init__(self, channels=16, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(N_CHANNELS, channels, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, 1))

    def forward(self, x):
        return self.body(x).squeeze(-1)


class CNN1DModel:
    needs_sequences = True

    def __init__(self, quantiles, seeds=(0, 1, 2), epochs=60, patience=8, lr=1e-3,
                 weight_decay=1e-3, val_frac=0.2, batch_size=64):
        self.quantiles = quantiles
        self.seeds, self.epochs, self.patience = seeds, epochs, patience
        self.lr, self.wd, self.val_frac, self.bs = lr, weight_decay, val_frac, batch_size

    def _train_one(self, seed, xtr, ytr, xva, yva):
        torch.manual_seed(seed)
        net = _Net()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(seed)
        for _ in range(self.epochs):
            net.train()
            for idx in torch.randperm(len(xtr), generator=g).split(self.bs):
                opt.zero_grad()
                loss_fn(net(xtr[idx]), ytr[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(xva), yva).item()
            if v < best - 1e-4:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net.eval()

    def fit(self, X, up, z, meta, seqs):
        self.base_ = BaselineModel(self.quantiles).fit(X, up, z)
        up = np.asarray(up, float)
        m = finite(up)
        dates = meta["date"].to_numpy()[m]
        x = torch.tensor(seqs[m])
        y = torch.tensor(up[m], dtype=torch.float32)
        ud = np.unique(dates)
        va = np.isin(dates, ud[int(len(ud) * (1 - self.val_frac)):])   # validation = ช่วงท้ายตามเวลา
        self.nets_ = [self._train_one(s, x[~va], y[~va], x[va], y[va]) for s in self.seeds]
        return self

    def predict(self, X, meta, seqs):
        x = torch.tensor(seqs)
        with torch.no_grad():
            p = np.mean([torch.sigmoid(n(x)).numpy() for n in self.nets_], axis=0)
        zq = self.base_.predict(X).drop(columns="p_up").to_numpy()
        return as_frame(p, zq, self.quantiles, X.index)
