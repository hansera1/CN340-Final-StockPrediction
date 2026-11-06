# Flow: CNN บนภาพ scalogram จากแท่ง 1 นาที (E19)

งานสำรวจ ไม่ใช้ตัดสินระบบส่งจริง · ลงทะเบียน: `results/e19_preregistration.md` · ผล: `results/e19_report.md`

```text
แท่ง 1m ภาคเช้า 10:00–12:30 (Settrade ก่อน, yfinance เมื่อไม่มี)
   ↓  1 ภาพต่อ (หุ้น, วัน) · แท่งจริง ≥ 120 จาก 150 นาที · ไม่ใช้แท่งหลัง 12:30
ภาพ 3 ช่อง 224×224 (r, |r|, log volume · Morlet ω₀ = 6, 32 scale, s = 2 ถึง n/2)
   ↓  experiments/27_e19_1m_scalogram.py → data/processed/e19_images.npz
ป้ายรอบบ่ายจาก rules.yaml (ราคาปิดทางการรวม ATC เทียบ P(14:00)) ขึ้น/ไม่ขึ้น
   ↓  experiments/28_e19_train.py · walk-forward ฝึกใหม่ทุกวัน · ขั้นต่ำ 20 วัน
Model A: ResNet18 / VGG16-BN / DenseNet121 × S0, S0-LR, S1, S2 → P(ขึ้น)
   ↓  out-of-fold ตัดตามวัน 4 fold
Model B: z = ln(ราคาปิดทางการ / P(c)) ด้วย LinearRegression บน P(ขึ้น) → ราคา = P(c)·exp(z)
   ↓
outputs/experiments/E19/predictions.csv + results/e19/metrics.csv
```

## ต่างจากโค้ดตั้งต้นของเพื่อน

| ตั้งต้น | ตอนนี้ | เหตุผล |
|---|---|---|
| หน้าต่างเลื่อน + horizon 15 นาที | 1 ตัวอย่างต่อ (หุ้น, วัน) | ตรงกับโจทย์ และไม่ทำให้ตัวอย่างซ้อนกัน |
| ป้าย: ราคาขึ้นใน 15 นาที | ป้ายรอบบ่ายจาก `rules.yaml` | เป้าหมายเดียวกับที่ส่งจริง |
| สร้าง `conv1` ใหม่ | ใช้น้ำหนัก ImageNet ครบ | เดิมทิ้งน้ำหนักชั้นแรก |
| แบ่ง train / valid / test ครั้งเดียว | walk-forward รายวัน | ตรงกับการใช้งานจริง |
| เลือก epoch จาก valid แล้ววัดบน valid | early stopping บนส่วนท้ายของข้อมูลฝึก | ไม่ให้ผลดีเกินจริง |
| Model B ฝึกจาก P(ขึ้น) in-sample | out-of-fold | ไม่ให้ Model B เชื่อ Model A เกินจริง |
| ทายราคาปิดเป็นบาท | ทาย z | ใช้กับหุ้นราคาต่างกันได้ |

## วิธีรัน

```powershell
python experiments/27_e19_1m_scalogram.py        # นับวันและสร้างภาพ
python experiments/28_e19_train.py               # ต้องมีวันทดสอบ ≥ 5 วัน ไม่งั้นรายงานว่าวัดผลไม่ได้
python experiments/28_e19_train.py --smoke 8     # ตรวจ pipeline เท่านั้น ห้ามอ้างเป็นผล
```

ไฟล์ CSV 1m จาก MT5 / SETSMART ใช้ได้ถ้าวางในรูปแบบเดียวกับ `data/raw/settrade/<SYMBOL>_1m.csv` และผ่านการเทียบกับ Settrade ก่อน
