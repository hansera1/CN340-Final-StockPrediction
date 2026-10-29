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
    return MODELS[name](quantiles, **kwargs)
