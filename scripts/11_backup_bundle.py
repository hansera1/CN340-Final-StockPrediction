"""งาน 9 ข้อ 8: แพ็กสิ่งที่เครื่องสำรองต้องใช้ (ไม่อยู่ใน git) เป็นไฟล์ zip เดียว — ไม่รวม .env

    python scripts/11_backup_bundle.py                 # → outputs/backup/backup_<เวลา>.zip
    (เครื่องสำรอง) python scripts/11_backup_bundle.py --restore <ไฟล์ zip>

รวม: outputs/models/ (ทุกโมเดลที่ล็อก + R4ATC + β ชั้น 2), data/raw/settrade + yfinance ของ KBANK / ADVANC,
     outputs/predictions/draft/, outputs/logs/predictions.jsonl, shadow_log.jsonl, evening_status_*.json
ทำหลัง refit 6 พ.ย. และทุกเย็นช่วงส่งจริง (ข้อมูลล่าสุด) แล้วคัดลอกไปเครื่องสำรอง
"""
import argparse
import zipfile
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, RAW_DIR, ROOT, load_rules


def files(rules):
    yield from (OUTPUT_DIR / "models").rglob("*")
    for src in ["settrade", "yfinance"]:
        for sym in rules["symbols"]:
            yield from (RAW_DIR / src).glob(f"{sym}_*.csv")
    yield from (OUTPUT_DIR / "predictions" / "draft").glob("*")
    for name in ["predictions.jsonl", "shadow_log.jsonl"]:
        yield OUTPUT_DIR / "logs" / name
    yield from (OUTPUT_DIR / "logs").glob("evening_status_*.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", type=Path, help="แตกไฟล์ zip ลงโปรเจกต์นี้ (เครื่องสำรอง)")
    args = ap.parse_args()
    if args.restore:
        with zipfile.ZipFile(args.restore) as z:
            bad = [n for n in z.namelist() if n.startswith(("/", "..")) or ".." in Path(n).parts or Path(n).name == ".env"]
            if bad:
                raise SystemExit(f"ไฟล์ zip มี path ที่ไม่ปลอดภัย: {bad[:3]}")
            z.extractall(ROOT)
            print(f"แตก {len(z.namelist())} ไฟล์ลง {ROOT} — ต่อด้วย python scripts/08_check_ready.py")
        return
    rules = load_rules()
    out = OUTPUT_DIR / "backup" / f"backup_{datetime.now():%Y%m%d_%H%M}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files(rules):
            if f.is_file() and f.name != ".env":
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"→ {out} ({n} ไฟล์, {out.stat().st_size / 1e6:.1f} MB) — ไม่มี .env (เครื่องสำรองกรอกเอง)")


if __name__ == "__main__":
    main()
