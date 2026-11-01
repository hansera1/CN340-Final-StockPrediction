"""สถานะงานเย็น (งาน 9 ข้อ 3) — outputs/logs/evening_status_<วันทำการที่เตรียมให้>.json

00_collect_daily.py เขียน: แต่ละแหล่งสำเร็จไหม, ร่างรอบเช้าสร้างไหม + model_version, ไฟล์ β ชั้น 2 สร้างไหม
05_predict_live.py morning อ่านก่อนรัน: ไม่มี / ไม่สำเร็จ → รันงานเย็นซ้ำทันที (--force --target-date วันนี้)
"""
import json

import pandas as pd

from forecast.config import OUTPUT_DIR
from forecast.live.log_schema import read_logs


def status_path(date):
    return OUTPUT_DIR / "logs" / f"evening_status_{pd.Timestamp(date):%Y-%m-%d}.json"


def draft_info(date):
    """→ (มีร่างไหม, model_version)"""
    path = OUTPUT_DIR / "predictions" / "draft" / f"{pd.Timestamp(date):%Y-%m-%d}_morning.jsonl"
    if not path.exists():
        return False, None
    rows = read_logs(path)
    return bool(rows), (rows[0]["model_version"] if rows else None)


def stage2_info(date):
    path = OUTPUT_DIR / "models" / "stage2" / f"{pd.Timestamp(date):%Y-%m-%d}.json"
    if not path.exists():
        return False, None
    d = json.loads(path.read_text(encoding="utf-8"))
    return True, d.get("reason")


def write_status(date, jobs, extra=None):
    """jobs: {ชื่องาน: True/False} → บันทึกสถานะรวม (ok = ทุกแหล่งสำเร็จ + มีร่าง + มีไฟล์ β)"""
    has_draft, mv = draft_info(date)
    has_beta, beta_reason = stage2_info(date)
    st = {"target_date": f"{pd.Timestamp(date):%Y-%m-%d}", "written_at": pd.Timestamp.now().isoformat(timespec="seconds"),
          "sources": jobs, "draft": {"ok": has_draft, "model_version": mv},
          "stage2": {"ok": has_beta, "reason": beta_reason}, **(extra or {})}
    st["ok"] = all(jobs.values()) and has_draft and has_beta
    path = status_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return st, path


def needs_rerun(date):
    """→ (ต้องรันงานเย็นซ้ำไหม, เหตุผล)"""
    path = status_path(date)
    if not path.exists():
        return True, f"ไม่มี {path.name}"
    st = json.loads(path.read_text(encoding="utf-8"))
    if not st.get("ok"):
        bad = [k for k, v in st.get("sources", {}).items() if not v]
        bad += [k for k in ("draft", "stage2") if not st.get(k, {}).get("ok")]
        return True, f"งานเย็นไม่สำเร็จ: {', '.join(bad) or 'ไม่ทราบ'}"
    return False, None
