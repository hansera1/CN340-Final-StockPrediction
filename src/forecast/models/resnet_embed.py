"""E7 ขั้น 0–1: ResNet18 pretrained ImageNet แบบ frozen → feature 512 มิติต่อแถว

- backbone อยู่ในโหมด eval ตลอด (BatchNorm ใช้ running statistics เดิม ไม่อัปเดต)
- สร้างภาพทีละ batch เพื่อไม่ให้ใช้หน่วยความจำเกิน; ใช้ GPU ถ้ามี
"""
import numpy as np
import torch

from forecast.features.images import build_images
from forecast.models.cnn1d import train_device


def load_backbone(device):
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval().to(device)


def embed_rows(bars, rows, rules, kind, length=56, batch=128):
    """→ (len(rows), 512) float32"""
    dev = train_device()
    net = load_backbone(dev)
    out = np.zeros((len(rows), 512), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            imgs = build_images(bars, rows.iloc[s:s + batch], rules, kind, length)
            out[s:s + batch] = net(torch.from_numpy(imgs).to(dev)).cpu().numpy()
    return out
