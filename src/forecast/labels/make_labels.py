"""สร้าง labels ต่อ (วัน, รอบ)

    r = log[ P(ปลายช่วง) / P(เริ่มช่วง) ]   → ทิศทาง ขึ้น / เท่าเดิม / ลง ตามโจทย์
    z = log[ P(ปลายช่วง) / P(c) ]           → เป้าหมายราคา เทียบราคาล่าสุดที่ทราบ ณ จุดตัด

P(เริ่มช่วง) ยังไม่ทราบตอนส่ง — ใช้เป็นคำตอบย้อนหลังเท่านั้น ห้ามนำไปเป็น feature

P(ปลายช่วง) = ราคาปิดของแท่งสุดท้ายที่จบ ณ/ก่อนเวลาปลายช่วง (ไม่ใช้ open ของแท่งประมูลที่เริ่ม 16:30)
p_close_auction = ราคาปิดของแท่งสุดท้ายหลัง 16:30 (ถ้าไฟล์มีรายการประมูลปิด)
flat = |P(ปลาย) − P(เริ่ม)| ≤ label.flat.ticks ช่องราคา (exact_equal = 0 ช่อง)
"""
import numpy as np
import pandas as pd

from forecast.config import session_times, to_local
from forecast.data.asof import last_close, price_asof, price_at_or_after, price_points
from forecast.evaluation.metrics import tick_size


def flat_ticks(rules):
    flat = rules["label"]["flat"]
    if flat["definition"] == "exact_equal":
        return 0
    if flat["definition"] == "within_ticks":
        return flat["ticks"]
    raise ValueError(f"label.flat.definition ไม่รู้จัก: {flat['definition']}")


def make_labels(bars, rules):
    lab = rules["label"]
    tol = pd.Timedelta(minutes=lab["start_price_tolerance_min"])
    stale = pd.Timedelta(minutes=lab["max_end_staleness_min"])
    bar = pd.Timedelta(minutes=rules["bar_minutes"])
    pm_start = to_local(rules["afternoon_history_start"], rules)
    symbol = bars["symbol"].iat[0]

    rows = []
    for date, day in bars.groupby("date", sort=True):
        pts = price_points(day)
        for sess in rules["sessions"]:
            if sess == "afternoon" and date < pm_start:
                continue
            t = session_times(rules, sess, date)

            # ปกติ = open ของแท่งที่เริ่ม ณ เวลานั้น; ถ้าจุดก่อนหน้าเก่ากว่า 1 แท่ง (ไม่มี open / พักเที่ยง)
            # ใช้ราคาแรกหลังเวลาเริ่มแทน
            p_start, start_time = price_asof(pts, t["start"])
            if pd.isna(start_time) or t["start"] - start_time >= bar:
                p_start, _ = price_at_or_after(pts, t["start"], tol)

            p_end, end_time = last_close(day, t["end"])
            if pd.isna(end_time) or t["end"] - end_time > stale:
                p_end = np.nan
            p_auction = np.nan
            if sess == "afternoon":
                after = day[day["bar_end"] > t["end"]]
                p_auction = float(after["close"].iat[-1]) if len(after) else np.nan
                if lab["price_1630"] == "closing_auction":
                    p_end = p_auction
                elif lab["price_1630"] != "last_trade_at_or_before":
                    raise ValueError(f"label.price_1630 ไม่รู้จัก: {lab['price_1630']}")

            p_ref, ref_time = last_close(bars, t["cutoff"])
            rows.append({
                "date": date, "symbol": symbol, "session": sess,
                "cutoff_ts": t["cutoff"], "p_start": p_start, "p_end": p_end,
                "p_ref": p_ref, "ref_time": ref_time,
                **({"p_close_auction": p_auction} if lab.get("store_closing_auction") else {}),
            })

    df = pd.DataFrame(rows)
    df["r"] = np.log(df["p_end"] / df["p_start"])
    df["z"] = np.log(df["p_end"] / df["p_ref"])
    # วัดเป็นช่องราคาเทียบราคาเริ่มช่วง ปัดเศษเพื่อกัน float error จาก CSV
    diff_ticks = np.round((df["p_end"] - df["p_start"]) / tick_size(df["p_start"]), 6)
    thr = flat_ticks(rules)
    df["direction"] = np.select([diff_ticks > thr, diff_ticks < -thr], [1, -1], 0).astype(float)
    df.loc[df["r"].isna(), "direction"] = np.nan
    # เป้าหมายของขั้นทิศทาง: P(ขึ้น) ของทุกกรณี (flat นับเป็น "ไม่ขึ้น") ไม่ใช่ P(ขึ้น | ไม่ flat)
    df["up"] = (df["direction"] == 1).astype(float).where(df["direction"].notna())
    return df
