"""โมเดล (หัวข้อ 07–08) — ทุกโมเดลมี fit(X, up, z, meta) / predict(X, meta) แบบเดียวกัน

predict คืน DataFrame: p_up + quantiles ของ z (q10, q25, q50, q75, q90)
"""
from forecast.models.baseline import BaselineModel
from forecast.models.cascade import CascadeModel
from forecast.models.independent import IndependentModel

MODELS = {
    "baseline": BaselineModel,
    "independent": IndependentModel,
    "cascade": CascadeModel,
}


def get_model(name, quantiles, **kwargs):
    if name == "cnn1d":   # import ช้า (torch) จึงโหลดเมื่อใช้เท่านั้น
        from forecast.models.cnn1d import CNN1DModel
        return CNN1DModel(quantiles, **kwargs)
    if name in ("e9_lr", "e9_gbm", "e9_zoo"):   # E9: ขึ้น vs ลง + ช่วงราคา conformal
        from forecast.models.e9 import E9GBM, E9LR, E9Zoo
        return {"e9_lr": E9LR, "e9_gbm": E9GBM, "e9_zoo": E9Zoo}[name](quantiles, **kwargs)
    if name == "finetune":   # E10: CNN pretrained แบบ fine-tune
        from forecast.models.finetune import FineTuneModel
        return FineTuneModel(quantiles, **kwargs)
    if name in ("frozen_lr", "frozen_head"):   # E7: หัวบน embedding ของ ResNet18
        from forecast.models.image_heads import FrozenHeadModel, FrozenLRModel
        return (FrozenLRModel if name == "frozen_lr" else FrozenHeadModel)(quantiles, **kwargs)
    return MODELS[name](quantiles, **kwargs)
