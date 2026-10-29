"""รูปแบบ log ของคำทำนาย — บันทึกก่อนรู้ผล ต่อท้ายไฟล์ JSONL ห้ามแก้ย้อนหลัง"""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class PredictionLog:
    prediction_id: str
    symbol: str
    session: str
    date: str
    data_cutoff: str
    source_received_at: str | None
    generated_at: str
    submitted_at: str | None
    model_version: str
    feature_version: str
    reference_price: float
    reference_time: str | None
    p_up: float
    direction: str
    z_quantiles: dict = field(default_factory=dict)      # q10..q90 ของ z
    price_quantiles: dict = field(default_factory=dict)  # ราคาปัดตามช่องราคาแล้ว
    point_price: float | None = None
    fallback_reason: str | None = None


def append_log(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")


def read_logs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
