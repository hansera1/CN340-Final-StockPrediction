"""ให้สคริปต์ใน scripts/ และ experiments/ import แพ็กเกจ forecast จาก src/ ได้โดยไม่ต้องติดตั้ง"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")   # ให้ภาษาไทยแสดงบน Windows console ได้

# งาน 9 ข้อ 1–2: พิมพ์/บันทึกเวลา + เขตเวลา, commit, ชื่อเครื่อง · เขตเวลาไม่ใช่ Asia/Bangkok → หยุด · นาฬิกาต่างเกิน 60 วินาที → เตือน
from forecast.runtime import startup  # noqa: E402

RUN = startup(Path(sys.argv[0]).stem)
