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
    data_source: str | None = None       # แหล่งแท่ง 1m วันนี้ (settrade / yfinance / history-only)
    data_lag_sec: float | None = None    # จุดตัด − เวลาจบของแท่ง 1m ล่าสุดที่ใช้
    data_age_min: float | None = None    # จุดตัด − เวลาจบของแท่งล่าสุดที่ใช้ทำ features (อายุข้อมูล)
    reference_source: str | None = None  # ที่มาของ P(c): Settrade 1d close (เช้า) / แท่ง 1m (บ่าย) / ทางสำรอง
    p_eq: float | None = None            # P(ราคาปลาย = P(c)) จาก A-price LR
    p_price: float | None = None         # P(ราคาปลาย > P(c) | ไม่เท่า)
    p_higher: float | None = None        # (1 − p_eq) × p_price
    p_lower: float | None = None


@dataclass
class ShadowLog:
    """รันเงา (E0-ud + ราคาล่าสุด ณ จุดตัด) — ใช้เทียบในรายงานเท่านั้น ห้ามส่ง ห้ามใช้เปลี่ยนโมเดล"""
    symbol: str
    session: str
    date: str
    data_cutoff: str
    generated_at: str
    model: str
    p_up: float
    direction: str
    reference_price: float
    point_price: float
    price_quantiles: dict = field(default_factory=dict)


def append_log(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")


def read_logs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
