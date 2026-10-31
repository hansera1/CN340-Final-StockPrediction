"""ส่งจริง: รันที่ c_เช้า / c_บ่าย แล้วส่งผลก่อน 09:30 / 12:30

    python scripts/05_predict_live.py morning
    python scripts/05_predict_live.py afternoon
    (ทดสอบย้อนหลัง: --date 2026-10-28 --now "2026-10-28 12:15" --no-fetch)

ก่อนทำนายจะดึงข้อมูลให้เอง (ปิดด้วย --no-fetch):
  - yfinance 60m 5 วันล่าสุด → ประวัติถึงเมื่อวาน; Settrade 1d → P(10:00) ของวันก่อน ๆ (ราคาประมูลเปิด)
  - แท่ง 1m วันนี้ทุกแหล่งใน live.sources (Settrade, yfinance) — แหล่งที่ล้มจะถูกข้าม

→ outputs/predictions/<วันที่>_<รอบ>.csv  (ไฟล์ส่งอาจารย์)
→ outputs/logs/predictions.jsonl            (log ก่อนรู้ผล ห้ามแก้)
→ outputs/logs/shadow_log.jsonl             (รันเงา E0-ud + ราคาล่าสุด — ห้ามส่ง ใช้เทียบในรายงานเท่านั้น)
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import INTERVAL_MIN, OUTPUT_DIR, load_rules, raw_source_dir
from forecast.data.io import drop_unfinished, merge_into_raw, raw_path
from forecast.live.log_schema import append_log
from forecast.live.predict import predict_session

SESSION_TH = {"morning": "เช้า", "afternoon": "บ่าย"}


def fetch_today(rules):
    tz = rules["timezone"]
    now = lambda: pd.Timestamp.now(tz=tz).tz_localize(None)
    iv = rules["data"]["interval"]
    jobs = {f"yfinance {iv}": lambda sym: ("yfinance", iv, _yf(sym, iv, "5d", rules)),
            "settrade 1d": lambda sym: ("settrade", "1d", _settrade_latest(sym, "1d", rules)),   # P(c) รอบเช้า
            "yfinance 1d": lambda sym: ("yfinance", "1d", _yf(sym, "1d", "5d", rules))}          # สำรองของ Settrade 1d
    for src in rules["live"]["sources"]:
        if src == "settrade":
            jobs["settrade 1m"] = lambda sym: ("settrade", "1m", _settrade_today(sym, rules))
        elif src == "yfinance":
            jobs["yfinance 1m"] = lambda sym: ("yfinance", "1m", _yf(sym, "1m", "1d", rules))
    for name, job in jobs.items():
        for sym in rules["symbols"]:
            try:
                src, iv, df = job(sym)
                df = drop_unfinished(df, rules, now(), INTERVAL_MIN[iv])
                added, _ = merge_into_raw(raw_path(sym, rules, raw_source_dir(rules, src), iv), df)
                last = df["timestamp"].max() if len(df) else "-"
                print(f"  ดึง {name} {sym}: +{added} แท่ง (ล่าสุด {last})")
            except BaseException as e:   # รวม SystemExit จาก .env ที่ไม่ครบ — ไม่ให้การดึงข้อมูลหยุดการทำนาย
                print(f"  ดึง {name} {sym}: ล้ม ({type(e).__name__}: {str(e)[:80]})")


def _yf(sym, interval, period, rules):
    from forecast.data.yfinance_src import fetch
    return fetch(sym, interval, rules, period)


_market = None


def _settrade():
    global _market
    from forecast.data.settrade import connect
    _market = _market or connect()
    return _market


def _settrade_today(sym, rules):
    from forecast.data.settrade import fetch_day
    return fetch_day(_settrade(), sym, pd.Timestamp.now(tz=rules["timezone"]).tz_localize(None).normalize(), rules)


def _settrade_latest(sym, interval, rules):
    from forecast.data.settrade import fetch_latest
    return fetch_latest(_settrade(), sym, interval, rules, limit=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", choices=["morning", "afternoon"])
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--now", help="จำลองเวลาปัจจุบัน (ทดสอบย้อนหลัง)")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--no-fetch", action="store_true", help="ไม่ดึงข้อมูลใหม่ ใช้ไฟล์ที่มีอยู่")
    args = ap.parse_args()
    rules = load_rules()
    t0 = time.perf_counter()
    if not args.no_fetch and not args.raw_dir:
        fetch_today(rules)

    records, shadows = predict_session(args.session, args.date, rules, args.raw_dir, now=args.now)
    # จำลองเวลา (--now) = ทดสอบย้อนหลัง/ซ้อม → log แยก ไม่ปนกับ log ส่งจริง
    logdir = OUTPUT_DIR / "logs" / ("dryrun" if args.now else "")
    for sh in shadows:
        append_log(logdir / "shadow_log.jsonl", sh)
    rows = []
    for r in records:
        r.submitted_at = str(datetime.now())
        append_log(logdir / "predictions.jsonl", r)
        p = r.price_quantiles
        rows.append({
            "วันที่": r.date, "รอบ": SESSION_TH[r.session], "หุ้น": r.symbol,
            "ทิศทาง": r.direction, "P(ขึ้น)": round(r.p_up, 3),
            "ราคาคาด": p["q50"], "ช่วง 50%": f"{p['q25']:.2f}–{p['q75']:.2f}",
            "ช่วง 80%": f"{p['q10']:.2f}–{p['q90']:.2f}",
            "ราคาอ้างอิง P(c)": r.reference_price, "ที่มา P(c)": r.reference_source,
            "P(ราคาเท่า P(c))": None if r.p_eq is None else round(r.p_eq, 3),
            "P(สูงกว่า)": None if r.p_higher is None else round(r.p_higher, 3),
            "P(ต่ำกว่า)": None if r.p_lower is None else round(r.p_lower, 3),
            "แหล่งข้อมูลวันนี้": r.data_source, "อายุข้อมูล (นาที)": None if r.data_age_min is None else round(r.data_age_min),
            "fallback": r.fallback_reason or "",
        })

    table = pd.DataFrame(rows)
    out = OUTPUT_DIR / "predictions" / ("dryrun" if args.now else "") / f"{args.date}_{args.session}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"\n→ {out}")
    run_sec = time.perf_counter() - t0
    log = OUTPUT_DIR / "logs" / "run_time.csv"
    pd.DataFrame([{"session": args.session, "date": args.date, "started": datetime.now(), "run_sec": run_sec,
                   "fetched": not args.no_fetch}]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"ใช้เวลา {run_sec:.1f} วินาที (ดึงข้อมูล + ทำนาย) → {log}")
    if table["fallback"].astype(bool).any():
        print("⚠ มีรายการที่ใช้ทางสำรอง — ตรวจสาเหตุในคอลัมน์ fallback")
    print(f"รันเงา {len(shadows)} รายการ → {logdir / 'shadow_log.jsonl'} (ไม่ส่ง)")


if __name__ == "__main__":
    main()
