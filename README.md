# พยากรณ์หุ้น KBANK และ ADVANC

โปรเจกต์ตาม **แผนพยากรณ์หุ้น ฉบับรวมสุดท้าย (29 ต.ค. 2026)** — โมเดลสองขั้น · ข้อมูลหลายความละเอียด · CNN และเวลา–ความถี่
ใช้งานจริง **9–13 พ.ย. 2026** ส่งคำทำนายให้อาจารย์ภายใน **09:30** (รอบเช้า) และ **12:30** (รอบบ่าย)

| รอบ | กำหนดส่ง | ทิศทาง | ราคาที่ต้องทำนาย |
|---|---|---|---|
| เช้า | 09:30 | P(12:30) เทียบ P(10:00) | P(12:30) |
| บ่าย | 12:30 | P(16:30) เทียบ P(14:00) | P(16:30) |

> สถานะ: โค้ดผ่านการทดสอบด้วย **ข้อมูลจำลอง** เท่านั้น ยังไม่ได้ตรวจกับข้อมูลจริง ไม่ใช่คำแนะนำการลงทุน

---

## โครงสร้างโฟลเดอร์

```
kbank_advanc_forecast/
│
├── configs/                  ⚙️  กติกากลาง — แก้ที่นี่ที่เดียว
│   ├── rules.yaml               เวลาตัด/ส่ง, นิยามราคา, flat, quantiles, walk-forward, โมเดลที่ใช้ส่งจริง
│   ├── experiments.yaml         E0–E6 จับคู่กับรหัสไอเดียเดิม (D, M, C)
│   └── feature_dictionary.csv   feature ทุกตัว + เวลาที่พร้อมใช้ + รอบที่ใช้ได้
│
├── data/                     📦  ข้อมูล (ไม่แก้ไฟล์ดิบ)
│   ├── raw/                     <source>/<SYMBOL>_<interval>.csv — yfinance/ (backtest 60m ~3 ปี ตั้งแต่ 24/10/2023), settrade/, _synthetic/
│   ├── external/                ตลาดต่างประเทศ ค่าเงิน พันธบัตร ฟลอว์ ปฏิทิน (มี available_at)
│   └── processed/               dataset ที่สร้างแล้ว (สร้างใหม่ได้เสมอ)
│
├── src/forecast/             🧠  โค้ดหลัก (แบ่งตามหน้าที่)
│   ├── config.py                โหลดกติกา + เวลาของแต่ละรอบ
│   ├── data/                    โหลด · ตรวจคุณภาพ · ราคา ณ จุดเวลา (as-of)
│   ├── labels/                  r, z, ทิศทาง, ราคาอ้างอิง P(c)
│   ├── features/                features ที่รู้ทันเวลา · ข้ามหุ้น · ข้อมูลเสริม · spectral · ลำดับสำหรับ CNN
│   ├── models/                  baseline (A) · independent (B) · cascade (C) · cnn1d (D)
│   ├── evaluation/              walk-forward · ตัวชี้วัด · block bootstrap · runner
│   └── live/                    ทำนายวันจริง · fallback · log
│
├── scripts/                  ▶️  รันตามลำดับเลข
│   ├── 00_collect_daily.py         สะสมข้อมูลทุกวันหลัง 17:00 (Settrade + yfinance, ไม่ซ้ำ)
│   ├── 00_fetch_yfinance.py        ดึงแท่ง 1d/60m/30m/15m/5m/1m จาก yfinance (+ วัดความหน่วง)
│   ├── 00_fetch_settrade.py        ดึงแท่งจาก Settrade Open API (+ วัดความหน่วง)
│   ├── 00_make_synthetic_data.py   ข้อมูลจำลองไว้ทดสอบ pipeline
│   ├── 01_check_data.py            ตรวจคุณภาพ + จำนวนตัวอย่างจริง
│   ├── 01b_validate_sources.py     yfinance 60m เทียบ Settrade 1m (ราคา label + features วันจริง)
│   ├── 02_build_dataset.py         สร้าง dataset
│   ├── 04_lock_model.py            ล็อกวิธี ฝึกโมเดลที่ล็อกทุกตัวใน rules.yaml → live
│   ├── 05_predict_live.py          ส่งจริง (เช้า/บ่าย)
│   ├── 06_score_live.py            ให้คะแนนหลังรู้ผล
│   └── 07_propose_cutoff.py        เสนอค่า c_เช้า / c_บ่าย จากความหน่วง + เวลารันที่วัดได้
│
├── experiments/              🧪  โค้ดการทดลอง (ไม่อยู่ในเส้นทางส่งจริง)
│   ├── 03_run_experiment.py        รัน E0–E12 + bootstrap (--holdout ใช้ครั้งเดียวหลังล็อก)
│   ├── 08 / 10 / 16_make_report*   รายงานรอบที่ 1 / 2 / 3 (PDF)
│   ├── 09_score_round2.py · 11_score_flat.py · 12_score_e12.py · 14_score_holdout.py
│   ├── 13_wavelet_coherence.py     A1 / A2 ความสัมพันธ์ในย่านความถี่สูง
│   └── 15_e11_pilot.py             E11 ResNet18 บน scalogram 5m (pilot)
│
├── results/                  📑  ผลทุกการทดลอง (lock_decision.md, holdout/, e12/, a1/, a2/, e11/, experiments/*.csv.gz)
│
├── tests/                    ✅  ทดสอบการรั่วของอนาคต · labels · โมเดล
│
├── docs/                     📄  สรุปแผนแยกหัวข้อ (อ่านง่าย)
│   ├── 01_โจทย์และกติกา.md
│   ├── 02_ข้อมูลและfeatures.md
│   ├── 03_โมเดลและการทดลอง.md
│   ├── 04_การประเมินผล.md
│   ├── 05_แผนเวลาและงาน.md
│   ├── 06_คู่มือวันจริง.md
│   └── 07_แหล่งอ้างอิง.md
│
└── outputs/                  📊  ผลลัพธ์ทั้งหมด (สร้างอัตโนมัติ)
    ├── reports/                 รายงานคุณภาพข้อมูล · คะแนนวันจริง
    ├── experiments/<E>/         summary.csv · bootstrap.csv · predictions ต่อ variant
    ├── models/<E>/<variant>/    โมเดลที่ล็อกแล้ว
    ├── predictions/             ไฟล์ส่งอาจารย์ <วันที่>_<รอบ>.csv
    └── logs/                    predictions.jsonl (บันทึกก่อนรู้ผล ห้ามแก้)
```

---

## เริ่มต้นใช้งาน

```bash
pip install -r requirements.txt

# 1) ทดสอบ pipeline ด้วยข้อมูลจำลอง
python scripts/00_make_synthetic_data.py
python experiments/03_run_experiment.py E0 E1 --raw-dir data/raw/_synthetic
python -m pytest -q

# 2) ข้อมูลจริง — backtest ใช้ yfinance 60m (~3 ปี ตั้งแต่ 24/10/2023, ขาด 2 วัน: 02/01/2024, 02/02/2026); Settrade 1m ย้อนได้แค่ ~4 วันทำการ
python scripts/00_fetch_yfinance.py                    # → data/raw/yfinance/<SYMBOL>_<interval>.csv
python scripts/00_fetch_settrade.py --start today      # (ต้องมี .env) → data/raw/settrade/<SYMBOL>_1m.csv
python scripts/01_check_data.py
python scripts/01b_validate_sources.py                 # ยืนยันว่า 60m ตรงกับรายนาที
python scripts/02_build_dataset.py
python experiments/03_run_experiment.py E0 E1 E2

# 3) ล็อกวิธี (โมเดลที่ล็อกอยู่ใน configs/rules.yaml → live; เหตุผลใน results/lock_decision.md)
python scripts/04_lock_model.py               # refit ศ. 6 พ.ย. ด้วยคำสั่งเดียวกัน

# 4) วันจริง (9–13 พ.ย.)
python scripts/05_predict_live.py morning     # ที่ c_เช้า ส่งก่อน 09:30
python scripts/05_predict_live.py afternoon   # ที่ c_บ่าย ส่งก่อน 12:30
python scripts/06_score_live.py               # ตอนเย็น
```

## ต้องยืนยันก่อนล็อกวิธี (แก้ใน `configs/rules.yaml`)

ค่าตอนนี้เป็นค่าชั่วคราว (มี comment "ชั่วคราว รอยืนยัน" ในไฟล์)

- [ ] `bar_timestamp` (ตอนนี้ `start`) — ดูจาก `01_check_data.py` ซึ่งพิมพ์แท่งแรก/สุดท้ายของแต่ละภาค แท่งหลัง 16:30 และเตือนถ้าไม่ตรง config
- [ ] `cutoff` ของแต่ละรอบ หลังวัดความหน่วงจริง (ตอนนี้ 09:15 / 12:15)
- [ ] `label.price_1630` (ตอนนี้ `last_trade_at_or_before`) — ราคาปิดประมูลเก็บแยกไว้ในคอลัมน์ `p_close_auction` สลับเป็น `closing_auction` ได้ทันที
- [ ] `label.flat.scoring` — ระหว่างรอ รายงานทั้ง `acc_flat_as_wrong` และ `acc_flat_excluded` (flat = ต่าง 0 tick)
- [ ] ขอบเขตโมเดลที่อาจารย์อนุญาต (รวม pretrained)
- [ ] รูปแบบไฟล์ส่งอาจารย์ และการปัด tick

## หลักที่โค้ดบังคับไว้

1. **ไม่เห็นอนาคต** — features ใช้เฉพาะแท่งที่ `bar_end ≤ cutoff`; วันจริงตัดข้อมูลดิบทิ้งก่อนเริ่ม (`tests/test_no_leakage.py`)
2. **backtest = วันจริง** — ทั้งสองใช้ `assemble_dataset()` ตัวเดียวกัน (ทดสอบว่าได้ค่าเท่ากัน)
3. **แบ่งตามเวลาเท่านั้น** — walk-forward รายสัปดาห์, กัน holdout ท้ายไว้ใช้หลังล็อก
4. **cascade ใช้ temporal OOF** — probability ขั้น 1 มาจากโมเดลที่ฝึกบนอดีตเท่านั้น
5. **เทียบโมเดลด้วย block bootstrap รายวัน** — ไม่ใช้ binomial เทียบ 50%
6. **มี fallback เสมอ** — ถ้าขั้นใดล้ม ใช้ baseline พร้อมธงเหตุผลใน log
