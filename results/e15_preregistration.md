# E15 — ลงทะเบียนเกณฑ์ล่วงหน้า (หลายโมเดล + feature selection + Optuna + ensemble, ทิศทาง)

**commit ก่อนรัน (7 พ.ย. 2026) · ห้ามแก้ส่วนบนหลังเห็นผล** · กติกา: `results/decision_rules.md` · โค้ด: `src/forecast/models/nested_tune.py`, `experiments/33_e15_models.py`

ข้อมูลเหมือน E14 · เป้าหมาย `up_notup` · pooled สองหุ้น + dummy หุ้น แยกรอบ

## โมเดลและ feature

- LR (L2), LR ElasticNet, LightGBM, XGBoost, Random Forest, SVM linear, MLP เล็ก (sklearn)
- F0 + F-daily (+ F-60m รอบบ่าย) + `flat_dir_features`

## Nested walk-forward

ฝึกขั้นต่ำ 120 วัน · fit ใหม่ทุก 5 วัน · จูนใหม่ทุก 120 วันทำการ · inner CV 4 fold แบบขยาย gap 5 วัน วัดด้วย log loss
ลำดับ feature: permutation importance บนส่วนท้าย 20% ของ inner-train ครั้งเดียวต่อ fold ต่อโมเดลด้วยค่าตั้งต้น · Optuna เลือก k ∈ {10, 15, 20, 30, ทั้งหมด} · ElasticNet ให้ L1 เลือกเอง · ลำดับ feature ที่ใช้ทำนายคำนวณใหม่ ณ วันที่จูน
Optuna: TPE, MedianPruner, seed 0, ≤ 25 trial ต่อโมเดลต่อครั้งที่จูน · ต้นไม้ลึก ≤ 4, min_data_in_leaf ≥ 30, learning rate ≤ 0.05 · MLP alpha ≥ 1, ≤ 2 ชั้น ชั้นละ ≤ 16 โหนด · LR/ElasticNet/SVM C log-uniform 0.001–1
ถ้าประเมินเวลาแล้วเกิน 12 ชม. ลดเหลือ ElasticNet + LightGBM + MLP

## Ensemble

- **ENS-2:** LR + MLP — **รายงานได้ ห้ามใช้ตัดสินในข้อ 9**
- **ENS-all:** เฉลี่ยทุกโมเดลเท่ากัน
- **ENS-w:** น้ำหนัก softmax(−inner-CV log loss) ของการจูนล่าสุด
- **ENS-top:** เฉลี่ยเฉพาะโมเดลที่ inner-CV ดีกว่า E0-un ในการจูนล่าสุด · ไม่มีเลย → ใช้ E0-un

## วัดผล

รายงาน: 3 กติกาวันนิ่ง, log loss, Brier เทียบ A-un, E0-un และ A-ud · CI Bonferroni **99.55%** (11 แบบ: 7 โมเดล + 4 ensemble) · แยกหุ้นและรอบ · PBO จาก CSCV ข้ามทุกแบบ (S = 10) · correlation ของความผิดพลาด · feature ที่ถูกเลือกบ่อยและความเสถียร · holdout รายงานเฉย ๆ · จำนวน trial ทั้งหมด

## ตัวแทนเข้าข้อ 9

ช่องละ 1 แบบ: inner-CV log loss เฉลี่ยต่ำที่สุด บนการจูนร่วมของแต่ละรอบ คำนวณจาก out-of-fold ของ best trial · ENS-2 ไม่เข้าแข่ง · ไม่ดูผลชุดทดสอบ
