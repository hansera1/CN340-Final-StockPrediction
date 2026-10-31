"""E10: CNN pretrained (ImageNet) แบบ fine-tune จริง — ขั้นบันได S0–S3 ต่อสถาปัตยกรรม

S0  backbone freeze → GAP feature → standardize + logistic (C = 0.01)   (เหมือน E7 ขั้น 0)
S1  backbone freeze → หัวใหม่ (GAP → dropout 0.5 → linear)              lr หัว 1e-3
S2  หัว + block สุดท้าย                                                  หัว 1e-3, block สุดท้าย 1e-4
S3  ทั้งโมเดล, layer-wise LR decay: หัว 1e-3 แล้ว × 0.3 ต่อกลุ่มชั้นไล่ไปถึงชั้นแรก

- เป้าหมาย: ขึ้น vs ลง บนวันที่ไม่ flat (เหมือน E9); ช่วงราคาใช้ conformal ของ E9 (เทียบกันได้)
- AdamW (weight decay 0.05 ใน S2–S3), warmup 2 epoch แล้ว cosine, ≤ 30 epoch, early stopping patience 5
  บน log loss ของ 20% วันท้ายของชุดฝึก; label smoothing 0.1; BatchNorm ใช้ running statistics เดิม (eval)
- bf16 autocast บน GPU; หลาย seed เฉลี่ยความน่าจะเป็น (ไม่เลือก seed ที่ดีที่สุด)
- ไม่มี augmentation (ห้าม flip / rotate / crop / color jitter ตามแผน)
- learning curve ราย epoch เก็บใน CURVES (03_run_experiment.py บันทึกเป็น <variant>_curves.csv)
"""
import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from forecast.models.cnn1d import train_device
from forecast.models.common import as_frame
from forecast.models.e9 import ConformalPrice, last_days_split, updown

CURVES = []
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _InceptionBody(nn.Module):
    """Inception-v3 ถึง Mixed_7c (ไม่ใช้ aux logits)"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        m = self.m
        x = m._transform_input(x)
        for name in ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1",
                     "Conv2d_4a_3x3", "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b",
                     "Mixed_6c", "Mixed_6d", "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c"]:
            x = getattr(m, name)(x)
        return x


def backbone(arch):
    """→ (body: ภาพ → feature map, มิติ feature, กลุ่มชั้นเรียงจากท้ายไปต้น, ขนาดภาพ input)
    กลุ่มแรก = "block สุดท้าย" ที่ปลดใน S2"""
    import torchvision.models as tv
    if arch == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        body = nn.Sequential(stem, m.layer1, m.layer2, m.layer3, m.layer4)
        return body, 512, [m.layer4, m.layer3, m.layer2, m.layer1, stem], 224
    if arch == "alexnet":
        f = tv.alexnet(weights=tv.AlexNet_Weights.IMAGENET1K_V1).features
        return f, 256, [f[10:], f[8:10], f[6:8], f[3:6], f[0:3]], 224
    if arch == "vgg16_bn":
        f = tv.vgg16_bn(weights=tv.VGG16_BN_Weights.IMAGENET1K_V1).features
        return f, 512, [f[34:], f[24:34], f[14:24], f[7:14], f[0:7]], 224
    if arch == "densenet121":
        f = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1).features
        body = nn.Sequential(f, nn.ReLU())
        return body, 1024, [nn.ModuleList([f.denseblock4, f.norm5]), nn.ModuleList([f.transition3, f.denseblock3]),
                            nn.ModuleList([f.transition2, f.denseblock2]),
                            nn.ModuleList([f.transition1, f.denseblock1]),
                            nn.ModuleList([f.conv0, f.norm0])], 224
    if arch == "inception_v3":
        m = tv.inception_v3(weights=tv.Inception_V3_Weights.IMAGENET1K_V1)
        m.aux_logits, m.AuxLogits = False, None
        groups = [nn.ModuleList([m.Mixed_7a, m.Mixed_7b, m.Mixed_7c]),
                  nn.ModuleList([m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e]),
                  nn.ModuleList([m.Mixed_5b, m.Mixed_5c, m.Mixed_5d]),
                  nn.ModuleList([m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3, m.Conv2d_3b_1x1,
                                 m.Conv2d_4a_3x3])]
        return _InceptionBody(m), 2048, groups, 299
    raise ValueError(f"ไม่รู้จัก arch: {arch}")


class _Net(nn.Module):
    def __init__(self, arch, dropout=0.5):
        super().__init__()
        self.body, dim, self.groups, self.size = backbone(arch)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def features(self, x):
        return self.head[1](self.head[0](self.body(x)))

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


_PRETRAINED = {}


def _fresh_net(arch, seed):
    """โหลดน้ำหนัก ImageNet ครั้งเดียวต่อ arch แล้วคัดลอก — หัว linear สุ่มใหม่ตาม seed ทุกครั้ง"""
    if arch not in _PRETRAINED:
        _PRETRAINED[arch] = _Net(arch)
    net = copy.deepcopy(_PRETRAINED[arch])
    torch.manual_seed(seed)
    net.head[3].reset_parameters()
    return net


def _prep(x_uint8, size, dev):
    """uint8 (n, 3, 224, 224) → float normalize แบบ ImageNet ขนาดตาม arch บน device"""
    x = x_uint8.to(dev, non_blocking=True).float().div_(255)
    x = (x - MEAN.to(dev)) / STD.to(dev)
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) if size != x.shape[-1] else x


def _freeze_bn(net):
    for mod in net.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


class FineTuneModel:
    needs_sequences = True

    def __init__(self, quantiles, arch="resnet18", stage="s2", seeds=(0, 1, 2), epochs=30, patience=5,
                 batch_size=64, lr_head=1e-3, lr_block=1e-4, layer_decay=0.3, weight_decay=0.05, warmup=2,
                 smoothing=0.1, val_frac=0.2, C=0.01, gpu_cache_bytes=2_500_000_000, accum_steps=1):
        self.quantiles, self.arch, self.stage, self.seeds = quantiles, arch, stage, seeds
        self.epochs, self.patience, self.bs = epochs, patience, batch_size
        self.lr_head, self.lr_block, self.layer_decay = lr_head, lr_block, layer_decay
        self.wd, self.warmup, self.smoothing, self.val_frac, self.C = weight_decay, warmup, smoothing, val_frac, C
        self.gpu_cache_bytes, self.accum_steps = gpu_cache_bytes, accum_steps

    # ---------- feature ของ backbone ที่ freeze (S0, S1) ----------
    @torch.no_grad()
    def _embed(self, net, imgs, dev, batch=128):
        net.eval()
        out = []
        for s in range(0, len(imgs), batch):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(net.features(_prep(imgs[s:s + batch], net.size, dev)).float().cpu())
        return torch.cat(out)

    def _param_groups(self, net):
        if self.stage == "s1":
            return [{"params": net.head.parameters(), "lr": self.lr_head, "weight_decay": 0.0}]
        groups = [{"params": net.head.parameters(), "lr": self.lr_head}]
        if self.stage == "s2":
            groups.append({"params": net.groups[0].parameters(), "lr": self.lr_block})
        else:   # s3: หัว 1e-3 แล้ว × layer_decay ต่อกลุ่ม
            for i, g in enumerate(net.groups):
                groups.append({"params": g.parameters(), "lr": self.lr_head * self.layer_decay ** (i + 1)})
        for g in groups:
            g["weight_decay"] = self.wd
        return groups

    def _train_one(self, seed, x_tr, y_tr, x_va, y_va, dev, fold):
        torch.manual_seed(seed)
        net = _fresh_net(self.arch, seed).to(dev)
        for p in net.parameters():
            p.requires_grad_(False)
        groups = self._param_groups(net)
        for g in groups:
            g["params"] = list(g["params"])
            for p in g["params"]:
                p.requires_grad_(True)
        opt = torch.optim.AdamW(groups)
        base = [g["lr"] for g in groups]
        n_tr = len(y_tr)
        steps_per_epoch = math.ceil(n_tr / self.bs)
        total, warm = self.epochs * steps_per_epoch, self.warmup * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        frozen_body = self.stage == "s1"
        if frozen_body:   # backbone ไม่เปลี่ยน → คำนวณ feature ครั้งเดียว ฝึกเฉพาะหัว
            f_tr, f_va = self._embed(net, x_tr, dev).to(dev), self._embed(net, x_va, dev).to(dev)
            head = net.head[2:]
        elif dev.type == "cuda" and x_tr.numel() + x_va.numel() < self.gpu_cache_bytes:
            # ย้ายภาพ uint8 ขึ้น GPU ครั้งเดียวต่อ fold — ค่าที่คำนวณเหมือนเดิม แค่ไม่ต้องคัดลอกทุก batch
            x_tr, x_va = x_tr.to(dev), x_va.to(dev)
        y_tr_s = (y_tr * (1 - self.smoothing) + 0.5 * self.smoothing).to(dev)
        y_va_d = y_va.to(dev)
        g = torch.Generator().manual_seed(seed)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            net.train()
            _freeze_bn(net)
            tl, tn = 0.0, 0
            for idx in torch.randperm(n_tr, generator=g).split(self.bs):
                opt.zero_grad(set_to_none=True)
                # แบ่ง batch เป็นส่วนย่อยแล้วสะสม gradient (BN ถูก freeze → gradient เท่ากับทั้ง batch) เพื่อลด VRAM
                loss_sum = 0.0
                for part in idx.split(math.ceil(len(idx) / self.accum_steps)):
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        logit = head(f_tr[part.to(dev)]).squeeze(-1) if frozen_body else \
                            net(_prep(x_tr[part.to(x_tr.device)], net.size, dev))
                    loss = F.binary_cross_entropy_with_logits(logit.float(), y_tr_s[part.to(dev)], reduction="sum")
                    (loss / len(idx)).backward()
                    loss_sum += loss.item()
                opt.step()
                sched.step()
                tl, tn = tl + loss_sum, tn + len(idx)
            net.eval()
            with torch.no_grad():
                logits = []
                for s in range(0, len(y_va), 128):
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        logits.append(head(f_va[s:s + 128]).squeeze(-1) if frozen_body else
                                      net(_prep(x_va[s:s + 128], net.size, dev)))
                v = F.binary_cross_entropy_with_logits(torch.cat(logits).float(), y_va_d).item()
            CURVES.append({"fold_train_end": fold, "seed": seed, "epoch": ep + 1, "train_loss": tl / tn,
                           "val_loss": v})
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net.cpu().eval()

    def fit(self, X, up, z, meta, imgs):
        t0 = time.perf_counter()
        self.price_ = ConformalPrice(self.quantiles).fit(meta)
        m, y = updown(meta)
        dates = meta["date"].to_numpy()[m]
        x = torch.from_numpy(np.ascontiguousarray(imgs[m]))
        y = torch.tensor(y, dtype=torch.float32)
        dev = train_device()
        fold = str(np.max(dates))[:10]
        if self.stage == "s0":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            net = _fresh_net(self.arch, 0).to(dev)
            emb = self._embed(net, x, dev).numpy()
            self.net_ = net.cpu()
            self.lr_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000)).fit(emb, y.numpy())
        else:
            tr, va = last_days_split(dates, self.val_frac)
            tr, va = torch.from_numpy(tr), torch.from_numpy(va)
            self.nets_ = [self._train_one(s, x[tr], y[tr], x[va], y[va], dev, fold) for s in self.seeds]
        self.fit_sec_ = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def _probs(self, net, imgs, dev):
        net.to(dev)
        logits = []
        for s in range(0, len(imgs), 128):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits.append(net(_prep(imgs[s:s + 128], net.size, dev)).float().cpu())
        net.cpu()
        return torch.sigmoid(torch.cat(logits)).numpy()

    def predict(self, X, meta, imgs):
        dev = train_device()
        x = torch.from_numpy(np.ascontiguousarray(imgs))
        if self.stage == "s0":
            emb = self._embed(self.net_.to(dev), x, dev).numpy()
            self.net_.cpu()
            ps = self.lr_.predict_proba(emb)[:, 1][None]
        else:
            ps = np.stack([self._probs(n, x, dev) for n in self.nets_])
        out = as_frame(ps.mean(axis=0), self.price_.predict(meta), self.quantiles, X.index)
        if len(ps) > 1:
            for i, p in enumerate(ps):
                out[f"p_up_seed{i}"] = np.clip(p, 1e-4, 1 - 1e-4)
        out["fit_sec"] = self.fit_sec_
        return out
