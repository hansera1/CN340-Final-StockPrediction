"""งาน 9 ข้อ 5: ตรวจความพร้อมก่อนส่ง — รันได้ทั้งเครื่องหลักและเครื่องสำรอง

    python scripts/08_check_ready.py
→ outputs/logs/check_ready_<เครื่อง>_<เวลา>.json และสำเนา results/round5/check_ready_<เครื่อง>.json
  (ไม่มีค่าใน .env · สำเนาใน results/ ซ่อนชื่อ Wi-Fi เพราะ repo เป็นสาธารณะ)

ตรวจ: เขตเวลา, นาฬิกา, .env ครบ (ชื่อเท่านั้น), ไฟล์โมเดลชุดที่ใช้ตาม rules.yaml, ข้อมูลล่าสุดถึงวันทำการก่อน,
ร่างรอบเช้า, ไฟล์ β ชั้น 2, เรียก Settrade API ได้ · มี FAIL → exit 1
"""
import json
import sys
from datetime import datetime

import _setup  # noqa: F401

from forecast.config import OUTPUT_DIR, ROOT, load_rules
from forecast.live.ready import run_checks
from forecast.runtime import public_network

MARK = {"PASS": "✔", "WARN": "!", "FAIL": "✘"}


def main():
    rules = load_rules()
    t, am, pm = run_checks(rules)
    print(f"\nความพร้อมของเครื่อง {_setup.RUN['host']} ({_setup.RUN['network']}) · commit {_setup.RUN['commit']}"
          f" · เตรียมรอบเช้า {am:%d/%m} / รอบบ่าย {pm:%d/%m}")
    for r in t.itertuples():
        print(f"  {MARK[r.status]} {r.status:4}  {r.check:38} {r.detail}")
    n_fail = int((t["status"] == "FAIL").sum())
    print(f"\nสรุป: {'พร้อม' if not n_fail else f'ไม่พร้อม ({n_fail} ข้อ FAIL)'}"
          + (f" · WARN {int((t['status'] == 'WARN').sum())} ข้อ" if (t["status"] == "WARN").any() else ""))
    rec = {"checked_at": datetime.now().isoformat(timespec="seconds"), "machine": _setup.RUN,
           "morning_target": f"{am:%Y-%m-%d}", "afternoon_target": f"{pm:%Y-%m-%d}", "ready": not n_fail,
           "checks": t.to_dict("records")}
    out = OUTPUT_DIR / "logs" / f"check_ready_{_setup.RUN['host']}_{datetime.now():%Y%m%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    # สำเนาที่ commit (repo สาธารณะ): ซ่อนชื่อ Wi-Fi
    pub = {**rec, "machine": {**rec["machine"], "network": public_network(rec["machine"]["network"])}}
    copy = ROOT / "results" / "round5" / f"check_ready_{_setup.RUN['host']}.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(json.dumps(pub, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}\n→ {copy}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
