# ข้อที่ต่างจาก docs/prompt_slides_final.md

ตัวเลขในสไลด์ใช้ค่าจากไฟล์ผลใน repo เสมอ เมื่อ prompt ต่างจากไฟล์ จะยึดไฟล์และบันทึกไว้ที่นี่

| # | ใน prompt | ที่ใช้ในสไลด์ | ที่มา |
|---|---|---|---|
| 1 | ค่าธรรมเนียม 0.157% รวม VAT | 0.157% + VAT 7% = **0.168% ต่อขา** | ตารางค่าธรรมเนียม settrade.com (broker 023) · results/financial_preregistration.md |
| 2 | ผลส่งจริงอยู่ที่ `outputs/logs/live_summary.csv` | VM เขียนไว้ที่ `outputs/reports/live_summary.csv` แล้วคัดลอกมาที่ `outputs/live_vm/` (gitignored) | scripts/06_score_live.py |
| 3 | `results/round7/` | ไม่มีโฟลเดอร์นี้ · รายงานรอบ 7 อยู่ที่ `docs/reports/report_round7.pdf` | — |
| 4 | walk-forward เริ่ม 24 เม.ย. 2567 | วันทดสอบแรกคือ **23 เม.ย. 2567** | dataset _wf |
| 5 | สูตร feature บางตัว | ใช้สูตรตามโค้ด: MACD และ ATR หารด้วย Close · `fd_bb_pos` = (C − ล่าง)/(บน − ล่าง) · `fd_ret_*` เป็น pct change ตัดปลาย ไม่ใช่ log return · `fd_oc_change` = (C − O)/O · A-ud ไม่มี feature ปฏิสัมพันธ์ | src/forecast/features/cn340.py, build.py |
| 6 | "38/38, 1,044" | ไม่พบไฟล์ใน repo ที่รองรับตัวเลขนี้ จึงไม่ใส่ | — |
| 7 | A2 "ราคาเปิดจาก Settrade ตรวจตรง 32/32" | ที่ตรวจคือ **ราคา 4 จุดของโจทย์ (ป้ายคำตอบ) จากแท่ง 60 นาทีเทียบ Settrade 1 นาที ตรง 32/32 จุด** | docs/02_ข้อมูลและfeatures.md |
| 8 | KBANK ช่อง 1 บาท | 105 แถวต่อรอบในข้อมูลทั้งหมด · **95 แถวต่อรอบในช่วงทดสอบ walk-forward** (สไลด์ใช้ทั้งสองค่าตามบริบท) | results/e14/kbank_flat_rate.csv |
| 9 | ไฟล์ `อ้างอิง.pptx` | ไม่พบใน repo · ทำรายการอ้างอิง APA ใหม่จากงานที่ใช้จริง | — |
| 10 | ตรวจ XIRR กับ Excel | Excel เรียกจากสคริปต์ไม่ได้ จึงตรวจด้วยตัวอย่างคำนวณมือใน tests/test_financial.py (ผ่าน 8/8) | tests/test_financial.py |
| 11 | Precision ของ E15 MLP รอบบ่าย "ไม่นิยาม" | E15 MLP ทาย "ขึ้น" 1 ครั้งและถูก → Precision = 1/1 = 100% (n = 1) · "ไม่นิยาม" ใช้กับ E0-un ที่ไม่ทาย "ขึ้น" เลย | outputs/experiments/E15/predictions.csv |
| 12 | ผลส่งจริง 9–13 พ.ย. | มีข้อมูลถึง **11 พ.ย. 2569 (3 วัน)** · 12–13 พ.ย. แสดง "รอข้อมูล" | outputs/live_vm/live_scored.csv |
| 13 | A-ud +4.4 จุดเทียบ E0-ud | ค่ามาจาก results/lock_decision.md (walk-forward รอบ 3 ป้ายชุดก่อน ATC รวมสองรอบ) ซึ่ง **ไม่ได้บันทึก n** · คำนวณซ้ำด้วยป้ายปัจจุบัน (C1235, n = 1,727 แถวที่ไม่นิ่ง) ได้ 53.2% จึงระบุในสไลด์ว่าเป็นป้ายชุดเดิม | results/lock_decision.md บรรทัด 27 |
| 14 | จำนวนวันข้อมูล 730 วัน | dataset _wf มี **729 วันทำการ** (24 ต.ค. 2566 – 29 ต.ค. 2569) · เอกสารเดิมเขียน 730 วัน ต่างกัน 1 วัน (ยังไม่ได้ตรวจสาเหตุ) | docs/slides/.build/slides_data.json (data.n_days) |

## ข้อจำกัดอื่นที่ระบุในสไลด์

- backtest ราคาไม่หักเงินปันผลวัน XD ขณะที่ระบบจริงหัก (สไลด์ 29)
- การเงินเป็นการจำลองย้อนหลัง ไม่รวม slippage ภาษี และค่าธรรมเนียมขั้นต่ำ (สไลด์ 30)
- ตาราง PBO ในสไลด์ 26 และภาคผนวก ค คัดจาก results/overfit_report.md, results/e20/pbo.csv และ results/e28_report.md โดยตรง ไม่ได้คำนวณใหม่
