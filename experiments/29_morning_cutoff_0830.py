"""รอบ 7 ข้อ 1.1: ย้ายจุดตัดเช้า 09:15 → 08:30 ต้องไม่เปลี่ยน dataset (feature รอบเช้าใช้ข้อมูลถึงวันก่อน)

โหลดฉบับเดิม (cache _pcoff_atc_c1235_wf = จุดตัดเช้า 09:15) เข้าหน่วยความจำก่อน แล้วสร้างใหม่ด้วย tag _wf_m0830 (ไม่ทับชื่อเดิม)
เทียบทุกแถว: คอลัมน์ f_ / fd_ / fh_ / fl_, p_ref, p_start, p_end, r, z, direction, up — ต่างแม้ค่าเดียว → หยุด

    python experiments/29_morning_cutoff_0830.py
"""
import importlib.util

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import PROCESSED_DIR, ROOT, load_rules
from forecast.evaluation.runner import label_tag, load_or_build_dataset

spec = importlib.util.spec_from_file_location("s17", ROOT / "experiments" / "17_label_variants.py")
s17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s17)
KEYS = ["date", "symbol", "session"]


def main():
    rules = load_rules()
    assert rules["sessions"]["morning"]["cutoff"] == "08:30"
    assert rules["label"]["price_1630"] == "closing_auction" and rules["sessions"]["afternoon"]["cutoff"] == "12:35"
    old_path = PROCESSED_DIR / f"dataset_yfinance_60m{label_tag(rules)}_wf.pkl"
    assert old_path.exists(), old_path
    old = pd.read_pickle(old_path)
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(rules["timezone"]) + pd.Timedelta(hours=23, minutes=59)
    new = load_or_build_dataset(rules, as_of=end, tag="_wf_m0830", rebuild=True)
    a, b = old.sort_values(KEYS).reset_index(drop=True), new.sort_values(KEYS).reset_index(drop=True)
    assert len(a) == len(b) and (a[KEYS].astype(str).to_numpy() == b[KEYS].astype(str).to_numpy()).all(), "แถวไม่ตรง"
    cols = [c for c in a if c.startswith(("f_", "fd_", "fh_", "fl_"))] + ["p_ref", "p_start", "p_end", "r", "z", "direction", "up"]
    assert set(cols) <= set(b.columns), set(cols) - set(b.columns)
    bad = [c for c in cols if not np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True)]
    assert not bad, f"ค่าเปลี่ยน: {bad}"
    am = a["session"] == "morning"
    moved = pd.to_datetime(b.loc[am, "cutoff_ts"]).dt.strftime("%H:%M").unique().tolist()
    print(f"เท่าเดิมทุกค่า: {len(b)} แถว ({am.sum()} เช้า), {len(cols)} คอลัมน์ · {b['date'].min().date()} – {b['date'].max().date()}"
          f" · จุดตัดเช้าใหม่ {moved}")


if __name__ == "__main__":
    main()
