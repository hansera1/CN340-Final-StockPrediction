# E17b — ลงทะเบียนเกณฑ์ล่วงหน้า (วิธีของ E15 กับโมเดลราคา)

**commit ก่อนรัน (7 พ.ย. 2026) · ห้ามแก้ส่วนบนหลังเห็นผล** · กติกา: `results/decision_rules.md` · โค้ด: `experiments/34_e17b_price_models.py`

แทน LR ใน A-price (`flat_price`) และ p_price (`price_updown`) ด้วย 7 โมเดลและ ensemble แบบ E15 · nested walk-forward, Optuna, feature selection เหมือน E15 · feature: ชุดเดิมของ A-price / p_price + F-daily + dummy หุ้น

ใช้กฎค่ามัธยฐาน และ PR-mean (ไม่ปัด) · ตัวเทียบ PR-A และ P(c)

## เกณฑ์ (รายงาน — ตัวตัดสินจริงคือข้อ 9)

CI Bonferroni **99.77%** (22 แบบ: 11 แบบ × กฎค่ามัธยฐาน / PR-mean) · ทั้งช่วงเป็นช่วงหลัก · เกณฑ์เหมือน E17a: MAE หรือ RMSE ต่ำกว่า PR-A และอีกตัวไม่แย่กว่า · ไม่แย่กว่าในแถวช่องราคา 1 บาท · coverage ห่างเป้าไม่มากกว่า PR-A

## ตัวแทนเข้าข้อ 9

ช่องละ 1 แบบ: inner-CV log loss รวม A-price + p_price เฉลี่ยต่ำที่สุด ไม่ดูผลชุดทดสอบ · ใช้คู่กับกฎค่ามัธยฐาน · ENS-2 ไม่เข้าแข่ง
