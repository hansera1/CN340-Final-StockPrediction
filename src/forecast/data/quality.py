"""ตรวจข้อมูลซ้ำ นาทีขาด ความผิดปกติ และ volume สะสม → รายงาน missingness รายวัน"""
import numpy as np
import pandas as pd

from forecast.config import session_times


def daily_quality(bars, rules):
    freq = (bars["bar_end"] - bars["bar_start"]).median()
    rows = []
    for date, day in bars.groupby("date", sort=True):
        row = {"date": date.date(), "n_bars": len(day)}
        for sess in rules["sessions"]:
            t = session_times(rules, sess, date)
            span = day[(day["bar_end"] > t["start"]) & (day["bar_end"] <= t["end"])]
            expected = max(1, int(round((t["end"] - t["start"]) / freq)))
            row[f"{sess}_bars"] = len(span)
            row[f"{sess}_missing_pct"] = round(max(0.0, 100 * (1 - len(span) / expected)), 1)
            row[f"{sess}_zero_volume"] = int((span["volume"] == 0).sum())
        row["nonpositive_price"] = int((day["close"] <= 0).sum())
        # ถ้า volume เพิ่มขึ้นเกือบทุกนาที น่าจะเป็นยอดสะสม ไม่ใช่รายนาที
        row["volume_looks_cumulative"] = bool((day["volume"].diff().dropna() >= 0).mean() > 0.95)
        row["max_gap_min"] = float(day["bar_end"].diff().dt.total_seconds().div(60).max() or np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_quality(q):
    cols = [c for c in q if c.endswith("_missing_pct")]
    return {
        "first_date": str(q["date"].min()),
        "last_date": str(q["date"].max()),
        "n_days": len(q),
        **{f"mean_{c}": round(q[c].mean(), 2) for c in cols},
        "days_cumulative_volume": int(q["volume_looks_cumulative"].sum()),
        "days_nonpositive_price": int((q["nonpositive_price"] > 0).sum()),
    }


def raw_timestamps(bars, rules):
    """timestamp ตามที่เขียนในไฟล์ดิบ (ก่อนแปลงเป็น bar_start / bar_end)"""
    return bars["bar_start"] if rules["bar_timestamp"] == "start" else bars["bar_end"]


def timestamp_profile(bars, rules):
    """เวลาแท่งแรก/สุดท้ายของแต่ละภาค และแท่งหลัง 16:30 — ใช้ยืนยัน bar_timestamp จากข้อมูลจริง

    แบ่งตามเวลาในไฟล์: เช้า < 13:00 ≤ บ่าย ≤ 16:30 < หลังปิด
    """
    ts = raw_timestamps(bars, rules)
    clock = ts.dt.strftime("%H:%M")
    day = ts.dt.date
    part = np.select([clock < "13:00", clock <= "16:30"], ["morning", "afternoon"], "after_1630")
    df = pd.DataFrame({"date": day, "clock": clock, "part": part})

    def mode_table(s):
        vc = s.value_counts()
        return ", ".join(f"{t} ({n} วัน)" for t, n in vc.head(3).items())

    sessions = df[df["part"] != "after_1630"].groupby(["date", "part"])["clock"]
    first, last = sessions.min().unstack(), sessions.max().unstack()
    after = df[df["part"] == "after_1630"]
    out = {}
    for p in ("morning", "afternoon"):
        if p in first:
            out[f"{p}_first_bar"] = mode_table(first[p].dropna())
            out[f"{p}_last_bar"] = mode_table(last[p].dropna())
    # ไม่ควรมีแท่ง continuous ที่ 12:30 / 16:30 (แท่งที่เริ่ม 12:30 = พักเที่ยง, 16:30 = ช่วงประมูลปิด)
    for hm in ("12:30", "16:30"):
        out[f"days_with_bar_at_{hm.replace(':', '')}"] = int(df.loc[df["clock"] == hm, "date"].nunique())
    out["days_with_bars_after_1630"] = f"{after['date'].nunique()} / {df['date'].nunique()}"
    out["times_after_1630"] = mode_table(after["clock"]) if len(after) else "-"

    # เดาจากแท่งสุดท้ายภาคเช้า (ไม่มีประมูลปิดมาปน): 12:30 = end; ก่อน 12:30 (12:29 ของ 1m, 12:00 ของ 60m) = start
    last_am = last["morning"].mode().iat[0] if "morning" in last and last["morning"].notna().any() else None
    guess = None if last_am is None else ("end" if last_am == "12:30" else "start" if last_am < "12:30" else None)
    out["bar_timestamp_guess"] = guess or f"ไม่แน่ใจ (แท่งสุดท้ายเช้า = {last_am})"
    out["bar_timestamp_config"] = rules["bar_timestamp"]
    if guess and guess != rules["bar_timestamp"]:
        out["WARNING"] = f"ข้อมูลดูเป็น '{guess}' แต่ rules.yaml ตั้ง '{rules['bar_timestamp']}' — แก้ก่อนรันต่อ"
    return out
