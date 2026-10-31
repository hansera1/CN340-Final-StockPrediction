"""วัดความหน่วงของแหล่งข้อมูลระหว่างเวลาตลาด — ใช้ตั้ง c_เช้า / c_บ่าย

ดึงแท่งล่าสุดถี่ ๆ (ทุก poll_sec วินาที) แล้วบันทึกว่าแท่งของแต่ละนาที "มาถึงครั้งแรกหลังนาทีนั้นจบ" ช้ากี่วินาที
    delay_sec = เวลาที่เห็นแท่งครั้งแรก (หลังเวลาจบแท่ง) − เวลาจบแท่ง     ความละเอียด ≈ poll_sec
แท่งที่ยังไม่จบไม่นับ; ถ้าค่าของแท่งเปลี่ยนหลังเห็นครั้งแรก จะนับเป็น revision
"""
import time

import pandas as pd


def measure_latency(fetch_last, rules, minutes, log_path, source, poll_sec=5, bar_minutes=1):
    """fetch_last() → DataFrame[timestamp, close, volume, ...] ของแท่งล่าสุดไม่กี่แท่ง (เวลาไทยไม่มี tz)"""
    tz = rules["timezone"]
    bar = pd.Timedelta(minutes=bar_minutes)
    now = lambda: pd.Timestamp.now(tz=tz).tz_localize(None)
    seen, rows, errors = {}, [], 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = now() + pd.Timedelta(minutes=minutes)
    while now() < stop:
        t0 = now()
        try:
            df = fetch_last()
            t1 = now()
            for r in df.itertuples(index=False):
                end = r.timestamp + (bar if rules["bar_timestamp"] == "start" else pd.Timedelta(0))
                if end > t1:
                    continue                                   # ยังไม่จบ
                key, val = r.timestamp, (r.close, r.volume)
                if key not in seen:
                    seen[key] = val
                    row = {"source": source, "bar_end": end, "first_seen": t1,
                           "delay_sec": (t1 - end).total_seconds(), "request_sec": (t1 - t0).total_seconds(),
                           "revised": False}
                    rows.append(row)
                    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
                    print(f"  แท่ง {end:%H:%M} มาถึง {t1:%H:%M:%S} ช้า {row['delay_sec']:.0f} วิ")
                elif seen[key] != val:                         # ค่าเปลี่ยนหลังเห็นครั้งแรก
                    seen[key] = val
                    for row in rows:
                        if row["bar_end"] == end:
                            row["revised"] = True
        except Exception as e:
            errors += 1
            print(f"  {now():%H:%M:%S} ล้ม: {str(e)[:120]}")
        time.sleep(max(0.0, poll_sec - (now() - t0).total_seconds()))
    out = pd.DataFrame(rows)
    out.attrs["errors"] = errors
    return out


def summarize_latency(df):
    if df.empty:
        return {"n_bars": 0, "errors": df.attrs.get("errors", 0)}
    # แท่งแรกที่เห็นตอนเริ่มวัดค้างมาก่อนแล้ว — ไม่ใช่ความหน่วงจริง
    d = df[df["first_seen"] > df["first_seen"].min()]
    return {
        "n_bars": len(d), "errors": df.attrs.get("errors", 0),
        "delay_median_sec": d["delay_sec"].median(), "delay_p95_sec": d["delay_sec"].quantile(0.95),
        "delay_max_sec": d["delay_sec"].max(), "revised_bars": int(d["revised"].sum()),
        "request_max_sec": d["request_sec"].max(),
    }


def propose_cutoff(deadline, max_delay_sec, run_sec, buffer_min=5):
    """c = เวลาส่ง − (ความหน่วงสูงสุด + เวลารัน + เผื่อส่งผล) ปัดลงเป็นนาที"""
    c = pd.Timestamp(f"2000-01-01 {deadline}") - pd.Timedelta(seconds=max_delay_sec + run_sec) \
        - pd.Timedelta(minutes=buffer_min)
    return c.floor("min").strftime("%H:%M")
