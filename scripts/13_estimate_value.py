"""เติมมูลค่าการซื้อขาย (value) ที่เป็น 0 ในไฟล์ Settrade — ไม่แก้ data/raw (ห้ามแก้ย้อนหลัง)

Settrade API ส่ง value (บาท = Σ ราคา × จำนวนหุ้นของทุกรายการในแท่ง) เฉพาะแท่งของวันที่ดึง
ดึงย้อนหลังได้ value = 0 เสมอ (ทดสอบ 2 พ.ย.: 29–30 ต.ค. ที่ในไฟล์มีค่า ดึงใหม่ได้ 0)
ไม่มีข้อมูลรายการซื้อขาย จึงประมาณด้วย value ≈ (open + high + low + close) / 4 × volume (ปัดเป็นบาท)
ตรวจกับแถวที่มี value จริง (29–30 ต.ค.): ผิดพลาดมัธยฐาน ≤ 0.09% ทุกความละเอียด (1m–1d)

    python scripts/13_estimate_value.py
→ data/processed/settrade_value/<SYMBOL>_<interval>.csv  รูปแบบเดียวกับไฟล์ดิบทุกตัวอักษร
  (คอลัมน์ ลำดับแถว รูปแบบตัวเลข CRLF) ต่างกันแค่ช่อง value ที่เป็น 0 และ volume > 0 ถูกแทนด้วยค่าประมาณ
→ ตารางตรวจความแม่น: data/processed/settrade_value/_check.csv
"""
import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import PROCESSED_DIR, RAW_DIR

SRC = RAW_DIR / "settrade"
OUT = PROCESSED_DIR / "settrade_value"
PRICE = ("open", "high", "low", "close")


def estimate(fields, col):
    return round(sum(float(fields[col[c]]) for c in PRICE) / 4 * float(fields[col["volume"]]))


def fill_file(src, dst):
    """แทนเฉพาะช่อง value ที่เป็น 0 → (จำนวนที่เติม, [(value จริง, ค่าประมาณ)] ไว้ตรวจความแม่น)"""
    with open(src, newline="", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    col = {name: i for i, name in enumerate(lines[0].rstrip("\r\n").split(","))}
    if "value" not in col:
        return None
    out, filled, pairs = [lines[0]], 0, []
    for line in lines[1:]:
        body = line.rstrip("\r\n")
        fields = body.split(",")
        if not body or any(fields[col[c]] == "" for c in (*PRICE, "volume", "value")):
            out.append(line)
            continue
        value, volume = float(fields[col["value"]]), float(fields[col["volume"]])
        if value > 0:
            pairs.append((value, estimate(fields, col)))
        elif volume > 0:
            fields[col["value"]] = f"{estimate(fields, col):.1f}"
            filled += 1
        out.append(",".join(fields) + line[len(body):])
    with open(dst, "w", newline="", encoding="utf-8") as f:
        f.writelines(out)
    return filled, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", help="ค่าเริ่มต้น: ทุกไฟล์ใน data/raw/settrade ที่มีคอลัมน์ value")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(SRC.glob("*.csv")):
        if args.symbols and path.stem.split("_")[0] not in args.symbols:
            continue
        res = fill_file(path, OUT / path.name)
        if res is None:
            continue
        filled, pairs = res
        print(f"{path.stem:12s} เติม {filled:6d} แถว · value จริง {len(pairs):5d} แถว")
        if pairs:
            real, est = np.array(pairs).T
            rel = np.abs(est - real) / real
            rows.append({"file": path.stem, "n_real": len(real), "median_abs_err_pct": np.median(rel) * 100,
                         "p95_abs_err_pct": np.quantile(rel, 0.95) * 100,
                         "total_err_pct": (est.sum() / real.sum() - 1) * 100})
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(OUT / "_check.csv", index=False)
        print("\nความแม่นของสูตรบนแถวที่มี value จริง:")
        print(t.round(3).to_string(index=False))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
