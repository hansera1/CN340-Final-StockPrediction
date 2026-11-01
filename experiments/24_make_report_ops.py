"""รายงานงาน 9: ระบบรันจริงและเครื่องสำรอง (HTML → PDF) — ข้อมูลจาก results/round5/

    python experiments/24_make_report_ops.py [--out outputs/reports/report_round5_ops.pdf]

ต้องรันก่อน: scripts/08_check_ready.py · คัดลอก notify_outbox / evening_status / รายการ Task Scheduler ลง results/round5/
"""
import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT

r5 = importlib.import_module("23_make_report_round5")
r2, r1 = r5.r2, r5.r1
esc, table = r1.esc, r1.table
RES = ROOT / "results" / "round5"
MARK = {"PASS": "✔ PASS", "WARN": "! WARN", "FAIL": "✘ FAIL"}


def ready_section():
    out = []
    for p in sorted(RES.glob("check_ready_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["machine"]
        rows = [[MARK[c["status"]], esc(c["check"]), esc(c["detail"])] for c in d["checks"]]
        out.append(f"<h3>เครื่อง {esc(m['host'])} ({esc(m['network'])}) · ตรวจเมื่อ {d['checked_at'].replace('T', ' ')} "
                   f"· commit {esc(m['commit'])} → {'พร้อม' if d['ready'] else 'ไม่พร้อม'}</h3>"
                   + table(["ผล", "ข้อที่ตรวจ", "รายละเอียด"], rows, "small"))
    return "".join(out)


def tasks_table():
    t = pd.read_csv(RES / "scheduled_tasks.csv")
    rows = [[esc(r.task), r.time, r.days, esc(r.action), "ใช่" if r.wake else "ไม่", "ใช่" if r.run_if_missed else "ไม่",
             r.next_run] for r in t.itertuples()]
    return table(["งาน", "เวลา", "วัน", "สคริปต์", "ปลุกเครื่อง", "รันทันทีถ้าพลาด", "รอบถัดไป"], rows, "small")


def evening_and_notify():
    st = json.loads((RES / "evening_status_2026-11-02.json").read_text(encoding="utf-8"))
    src = ", ".join(f"{k} {'✔' if v else '✘'}" for k, v in st["sources"].items())
    ev = table(["วันที่เตรียมให้", "แหล่งข้อมูล", "ร่างรอบเช้า (model_version)", "β ชั้น 2", "สถานะรวม"],
               [[st["target_date"], esc(src), esc(st["draft"]["model_version"]), "มี" if st["stage2"]["ok"] else "ไม่มี",
                 "สำเร็จ" if st["ok"] else "ไม่สำเร็จ"]], "small")
    msgs = [json.loads(x) for x in (RES / "notify_samples.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    nt = table(["เวลา", "ประเภท", "ข้อความ", "ผลการส่ง"],
               [[m["at"].replace("T", " "), m["kind"], esc(m["text"]).replace("\n", "<br>"), esc(m["sent"])] for m in msgs],
               "small")
    return ev, nt


def build_html(nar):
    ev, nt = evening_and_notify()
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><title>รายงานงาน 9</title>
<style>{r2.CSS}</style></head><body>
<h1>งาน 9: ระบบรันจริงและเครื่องสำรอง</h1>
<div class="sub">KBANK / ADVANC · ทำวันที่ 1 พ.ย. 2026 (ก่อนซ้อมรอบ 1 วันที่ 3 พ.ย.) · สร้างเมื่อ {datetime.now():%d/%m/%Y %H:%M}</div>
{nar["summary"]}

<h2>1. สิ่งที่ทำ (ข้อ 1–8)</h2>
{nar["done"]}

<h2 class="pb">2. ผลตรวจความพร้อม (ข้อ 5 — 08_check_ready.py)</h2>
{ready_section()}
{nar["ready"]}

<h2>3. งานตั้งเวลาอัตโนมัติ (ข้อ 6)</h2>
{tasks_table()}
{nar["tasks"]}

<h2 class="pb">4. สถานะงานเย็นและตัวอย่างข้อความแจ้งเตือน (ข้อ 3–4)</h2>
{ev}
{nt}
{nar["notify"]}

<h2>5. Test (เกณฑ์เสร็จข้อ 1–5)</h2>
{nar["tests"]}

<h2 class="pb">6. ขั้นตอนเครื่องสำรอง (ข้อ 8)</h2>
{nar["backup"]}

<h2>7. สิ่งที่กลุ่มต้องทำต่อ</h2>
{nar["todo"]}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "reports" / "report_round5_ops.pdf")
    ap.add_argument("--narrative", type=Path, default=ROOT / "results" / "round5_ops_narrative.py")
    args = ap.parse_args()
    ns = {}
    exec(args.narrative.read_text(encoding="utf-8"), ns)
    src = args.out.with_suffix(".html")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(build_html(ns["NARRATIVE"]), encoding="utf-8")
    r5._render(src, args.out)
    print("→", args.out)


if __name__ == "__main__":
    main()
