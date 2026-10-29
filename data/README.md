# data/

| โฟลเดอร์ | เก็บอะไร | แก้ได้ไหม |
|---|---|---|
| `raw/` | แท่งรายนาที `<SYMBOL>_1min.csv` จากแหล่งข้อมูล | ❌ ห้ามแก้ — เพิ่มข้อมูลใหม่ต่อท้ายเท่านั้น |
| `raw/_synthetic/` | ข้อมูลจำลองจาก `00_make_synthetic_data.py` | ใช้ทดสอบ pipeline เท่านั้น |
| `external/` | ข้อมูลเสริม (ต่างประเทศ ค่าเงิน พันธบัตร ฟลอว์ ปฏิทิน) | เพิ่มไฟล์ได้ |
| `processed/` | dataset ที่สร้างแล้ว | ลบได้ — สร้างใหม่ด้วย `02_build_dataset.py` |

## รูปแบบ `raw/<SYMBOL>_1min.csv`

| คอลัมน์ | จำเป็น | หมายเหตุ |
|---|---|---|
| `timestamp` | ✅ | เวลาไทย; ระบุใน `rules.yaml → bar_timestamp` ว่าเป็นเวลาเริ่มหรือจบแท่ง |
| `close` | ✅ | |
| `volume` | ✅ | ต้องเป็นรายนาที ไม่ใช่ยอดสะสม (`01_check_data.py` ตรวจให้) |
| `open`, `high`, `low` | แนะนำ | ไม่มี `open` → P(10:00)/P(14:00) ใช้ราคาแรกหลังเวลาเริ่ม |
| `value` | ไม่จำเป็น | |

```csv
timestamp,open,high,low,close,volume
2026-10-28 10:01:00,152.0,152.5,151.5,152.5,120400
```

## รูปแบบ `external/<ชื่อ>.csv`

คอลัมน์ `available_at` = **เวลาที่ค่าพร้อมใช้จริง** (ไม่ใช่เวลาของเหตุการณ์) + คอลัมน์ตัวเลขใดก็ได้
→ กลายเป็น feature `f_ext_<ชื่อ>_<คอลัมน์>` โดยใช้ค่าล่าสุดที่ `available_at ≤ cutoff`

```csv
available_at,spx_ret,ndx_ret
2026-10-28 04:00:00,0.0042,0.0061
```

ไฟล์ที่วางแผนไว้: `us.csv`, `asia.csv`, `fx.csv`, `bond.csv`, `flow.csv`, `tfex.csv`, `calendar.csv` (ดู `configs/feature_dictionary.csv`)
