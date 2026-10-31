"""ข้อมูลเสริมจาก data/external/*.csv — join แบบ as-of ด้วยเวลาที่ค่าพร้อมใช้จริง

รูปแบบไฟล์: available_at (เวลาที่ค่าเผยแพร่/ได้รับจริง, Asia/Bangkok) + คอลัมน์ตัวเลข
เช่น us.csv: available_at, spx_ret, ndx_ret
แต่ละแถวของ dataset จะได้ค่าล่าสุดที่ available_at <= cutoff_ts เท่านั้น
calendar.csv (วันหยุด / XD) ไม่ใช่ feature → ข้าม
"""
import pandas as pd

from forecast.config import EXTERNAL_DIR

NOT_FEATURES = {"calendar"}


def load_external(rules, ext_dir=None):
    tables = {}
    for path in sorted((ext_dir or EXTERNAL_DIR).glob("*.csv")):
        if path.stem in NOT_FEATURES:
            continue
        df = pd.read_csv(path)
        ts = pd.to_datetime(df["available_at"])
        tz = rules["timezone"]
        df["available_at"] = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
        tables[path.stem] = df.sort_values("available_at")
    return tables


def asof_join(rows, ext, name):
    values = [c for c in ext if c != "available_at"]
    ext = ext.rename(columns={c: f"f_ext_{name}_{c}" for c in values})
    ext[f"avail_{name}"] = ext["available_at"]
    out = pd.merge_asof(rows.sort_values("cutoff_ts"), ext, left_on="cutoff_ts",
                        right_on="available_at", direction="backward")
    out[f"f_ext_{name}_age_min"] = (out["cutoff_ts"] - out[f"avail_{name}"]).dt.total_seconds() / 60
    return out.drop(columns=["available_at", f"avail_{name}"])
