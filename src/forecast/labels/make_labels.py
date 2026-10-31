"""สร้าง labels ต่อ (วัน, รอบ)

    r = log[ P(ปลายช่วง) / P(เริ่มช่วง) ]   → ทิศทาง ขึ้น / เท่าเดิม / ลง ตามโจทย์
    z = log[ P(ปลายช่วง) / P(c) ]           → เป้าหมายราคา เทียบราคาล่าสุดที่ทราบ ณ จุดตัด

P(เริ่มช่วง) ยังไม่ทราบตอนส่ง — ใช้เป็นคำตอบย้อนหลังเท่านั้น ห้ามนำไปเป็น feature

P(10:00), P(14:00) = open ของแท่งแรกของภาค (แท่งประมูลเปิด 09:55–10:00 / 13:55–14:00 ถ้าไฟล์มี)
    ภาคเช้า = แท่งที่เริ่มก่อน 12:30, ภาคบ่าย = แท่งที่เริ่ม 12:30–16:30; แท่งแรกต้องเริ่มไม่เกิน
    เวลาเริ่มช่วง + start_price_tolerance_min ไม่งั้นถือว่าข้อมูลหาย
P(ปลายช่วง) = ราคาปิดของแท่งสุดท้ายที่จบ ณ/ก่อนเวลาปลายช่วง (1m: แท่ง ≤ 12:29 / ≤ 16:29) ไม่รวมแท่งประมูลปิด
    นาทีที่ไม่มีการซื้อขายไม่มีแท่ง → ใช้ "แท่งล่าสุดที่ ≤ เวลานั้น" เสมอ
p_close_auction = ราคาปิดของแท่งสุดท้ายหลัง 16:30 (ถ้าไฟล์มีรายการประมูลปิด)
flat = |P(ปลาย) − P(เริ่ม)| ≤ label.flat.ticks ช่องราคา (exact_equal = 0 ช่อง)
"""
import numpy as np
import pandas as pd

from forecast.config import raw_source_dir, session_times, to_local
from forecast.data.asof import last_close
from forecast.evaluation.metrics import tick_size


def flat_ticks(rules):
    flat = rules["label"]["flat"]
    if flat["definition"] == "exact_equal":
        return 0
    if flat["definition"] == "within_ticks":
        return flat["ticks"]
    raise ValueError(f"label.flat.definition ไม่รู้จัก: {flat['definition']}")


def first_bar_open(day, rules, sess, t, tol):
    """open ของแท่งแรกของภาค (แท่งประมูลเปิด) — ไม่มีคอลัมน์ open ใช้ close ของแท่งนั้น"""
    lo = t["start"].normalize() if sess == "morning" else session_times(rules, "morning", t["start"])["end"]
    hi = t["end"]
    b = day[(day["bar_start"] >= lo) & (day["bar_start"] < hi)]
    if b.empty or b["bar_start"].iat[0] > t["start"] + tol:
        return np.nan
    col = "open" if "open" in b and pd.notna(b["open"].iat[0]) else "close"
    return float(b[col].iat[0])


def make_labels(bars, rules):
    lab = rules["label"]
    tol = pd.Timedelta(minutes=lab["start_price_tolerance_min"])
    stale = pd.Timedelta(minutes=lab["max_end_staleness_min"])
    pm_start = to_local(rules["afternoon_history_start"], rules)
    symbol = bars["symbol"].iat[0]

    rows = []
    for date, day in bars.groupby("date", sort=True):
        for sess in rules["sessions"]:
            if sess == "afternoon" and date < pm_start:
                continue
            t = session_times(rules, sess, date)

            p_start = first_bar_open(day, rules, sess, t, tol)

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

    return finalize_labels(pd.DataFrame(rows), rules)


def finalize_labels(df, rules):
    """r, z, ทิศทาง 3 สถานะ และ up จาก p_start / p_end / p_ref"""
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


def daily_opens(symbol, rules):
    """open รายวัน (= ราคาประมูลเปิดเช้า, ตรวจกับ Settrade 1m แล้ว) จาก data.daily_open_sources ตามลำดับ"""
    opens = None
    for src in rules["data"].get("daily_open_sources", []):
        path = raw_source_dir(rules, src) / f"{symbol}_1d.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, parse_dates=["timestamp"])
        s = pd.Series(d["open"].to_numpy(), index=d["timestamp"].dt.normalize())
        s = s[~s.index.duplicated(keep="last")]
        opens = s if opens is None else opens.combine_first(s)
    return opens


def labels_for(bars, rules, interval):
    """labels ของไฟล์ตาม interval — ถ้าไม่ใช่ 1m ใช้ open รายวันเป็น P(10:00) แทน open ของแท่งหยาบ
    (yfinance 60m: open แท่ง 10:00 = ราคา ณ 10:00 ไม่ใช่ราคาประมูลเปิด ต่างกัน 1 tick ใน 3/8 วันที่ตรวจ)"""
    lab = make_labels(bars, rules)
    lab["p_start_source"] = "bar"
    if interval == "1m" or lab.empty:
        return lab
    opens = daily_opens(bars["symbol"].iat[0], rules)
    if opens is None:
        return lab
    morning = lab["session"] == "morning"
    o = lab["date"].dt.tz_localize(None).map(opens)
    use = morning & o.notna()
    lab.loc[use, "p_start"] = o[use]
    lab.loc[use, "p_start_source"] = "daily_open"
    return finalize_labels(lab, rules)
