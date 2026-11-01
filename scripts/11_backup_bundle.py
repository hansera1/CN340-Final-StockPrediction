"""งาน 9 ข้อ 8: แพ็กสิ่งที่อีกเครื่องต้องใช้ (ไม่อยู่ใน git) เป็นไฟล์ zip เดียว — ไม่รวม .env

    python scripts/11_backup_bundle.py                     # ทุกอย่าง → outputs/backup/backup_<เครื่อง>_<เวลา>.zip
    python scripts/11_backup_bundle.py --models-only       # เฉพาะโมเดล (ซิงก์หลัง refit 6 พ.ย.)
    (อีกเครื่อง) python scripts/11_backup_bundle.py --restore <ไฟล์ zip>

รวม: outputs/models/ (โมเดลที่ล็อก + R4ATC · ไม่รวม β ชั้น 2 ซึ่งแต่ละเครื่องฝึกเองตอนเย็น), data/raw/settrade + yfinance ของ KBANK / ADVANC,
     outputs/predictions/draft/, outputs/logs/predictions.jsonl, shadow_log.jsonl, evening_status_*.json
restore: โมเดลและข้อมูลดิบแตกลงที่เดิม · log / ร่าง / สถานะงานเย็นของเครื่องต้นทางไปที่ outputs/logs/from_<เครื่องต้นทาง>/
         (log สองเครื่องแยกกัน ไม่เขียนทับ — กติกาสองเครื่องใน docs/06_คู่มือวันจริง.md)
"""
import argparse
import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, RAW_DIR, ROOT, load_rules

SEPARATE = ("outputs/logs/", "outputs/predictions/draft/")   # ของเครื่องต้นทาง → from_<host>/ ไม่ทับของเครื่องนี้


def files(rules, models_only):
    # β ชั้น 2 ไม่ซิงก์: แต่ละเครื่องฝึกเองทุกเย็นจาก log ของตัวเอง
    yield from (f for f in (OUTPUT_DIR / "models").rglob("*") if "stage2" not in f.parts)
    if models_only:
        return
    for src in ["settrade", "yfinance"]:
        for sym in rules["symbols"]:
            yield from (RAW_DIR / src).glob(f"{sym}_*.csv")
    yield from (OUTPUT_DIR / "predictions" / "draft").glob("*")
    for name in ["predictions.jsonl", "shadow_log.jsonl"]:
        yield OUTPUT_DIR / "logs" / name
    yield from (OUTPUT_DIR / "logs").glob("evening_status_*.json")


def restore(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts or Path(n).name == ".env"]
        if bad:
            raise SystemExit(f"ไฟล์ zip มี path ที่ไม่ปลอดภัย: {bad[:3]}")
        meta = json.loads(z.read("bundle.json")) if "bundle.json" in names else {"host": "unknown"}
        sep = OUTPUT_DIR / "logs" / f"from_{meta['host']}"
        n_main = n_sep = 0
        for n in names:
            if n == "bundle.json" or n.endswith("/"):
                continue
            if n.startswith(SEPARATE):
                dest = sep / Path(n).relative_to(*Path(n).parts[:2])
                n_sep += 1
            else:
                dest = ROOT / n
                n_main += 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))
    print(f"จาก {meta['host']} ({meta.get('created')}, {meta.get('kind')}): แตก {n_main} ไฟล์ลงโปรเจกต์"
          + (f", log/ร่างของเครื่องต้นทาง {n_sep} ไฟล์ → {sep}" if n_sep else "")
          + "\nต่อด้วย: python scripts/08_check_ready.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", type=Path, help="แตกไฟล์ zip ลงโปรเจกต์นี้ (อีกเครื่อง)")
    ap.add_argument("--models-only", action="store_true", help="เฉพาะ outputs/models/ (ซิงก์หลัง refit)")
    args = ap.parse_args()
    if args.restore:
        return restore(args.restore)
    rules = load_rules()
    host = platform.node()
    kind = "models" if args.models_only else "full"
    out = OUTPUT_DIR / "backup" / f"backup_{kind}_{host}_{datetime.now():%Y%m%d_%H%M}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bundle.json", json.dumps({"host": host, "created": datetime.now().isoformat(timespec="seconds"),
                                              "kind": kind, "commit": _setup.RUN["commit"]}))
        for f in files(rules, args.models_only):
            if f.is_file() and f.name != ".env":
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"→ {out} ({n} ไฟล์, {out.stat().st_size / 1e6:.1f} MB) — ไม่มี .env (อีกเครื่องกรอกเอง)")


if __name__ == "__main__":
    main()
