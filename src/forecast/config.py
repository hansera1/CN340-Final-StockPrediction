"""โหลดกติกากลางและพาธมาตรฐานของโปรเจกต์"""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"


INTERVAL_MIN = {"1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "60m": 60,
                "120m": 120, "240m": 240, "1d": 1440}


def interval_minutes(interval):
    return INTERVAL_MIN[interval]


def raw_source_dir(rules, source=None):
    """data/raw/<source>/ — source ตั้งต้นจาก rules.yaml → data.source"""
    return RAW_DIR / (source or rules["data"]["source"])


def load_yaml(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rules():
    return load_yaml("rules.yaml")


def load_experiments():
    return load_yaml("experiments.yaml")


def to_local(ts, rules):
    ts = pd.Timestamp(ts)
    tz = rules["timezone"]
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def session_times(rules, session, date):
    """เวลาตัดข้อมูล กำหนดส่ง และจุดเวลาเป้าหมายของรอบหนึ่งในวันหนึ่ง (Asia/Bangkok)"""
    day = to_local(date, rules).normalize()
    cfg = rules["sessions"][session]

    def at(hhmm):
        h, m = hhmm.split(":")
        return day + pd.Timedelta(hours=int(h), minutes=int(m))

    return {
        "cutoff": at(cfg["cutoff"]),
        "deadline": at(cfg["submit_deadline"]),
        "start": at(cfg["start_time"]),
        "end": at(cfg["end_time"]),
    }


def quantile_cols(rules):
    return [f"q{round(q * 100):02d}" for q in rules["quantiles"]]
