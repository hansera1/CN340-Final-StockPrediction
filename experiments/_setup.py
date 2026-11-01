"""ให้สคริปต์ใน scripts/ และ experiments/ import แพ็กเกจ forecast จาก src/ ได้โดยไม่ต้องติดตั้ง"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")   # ให้ภาษาไทยแสดงบน Windows console ได้
