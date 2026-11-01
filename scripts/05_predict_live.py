"""ส่งจริง: รันที่ c_เช้า / c_บ่าย แล้วส่งผลก่อน 09:30 / 12:30

    python scripts/05_predict_live.py morning
    python scripts/05_predict_live.py afternoon
    (ทดสอบย้อนหลัง: --date 2026-10-28 --now "2026-10-28 12:15" --no-fetch)
    (ซ้อมกับสำเนาข้อมูล: --data-root <โฟลเดอร์ที่มี settrade/ yfinance/> --no-fetch --now ...)

ก่อนทำนายจะดึงข้อมูลให้เอง (ปิดด้วย --no-fetch):
  - yfinance 60m 5 วันล่าสุด → ประวัติถึงเมื่อวาน; Settrade 1d → P(10:00) ของวันก่อน ๆ (ราคาประมูลเปิด)
  - แท่ง 1m วันนี้ทุกแหล่งใน live.sources (Settrade หลัก, yfinance สำรอง) — แหล่งที่ล้มจะถูกข้าม

ก่อนทำนายตรวจข้อมูลรายวันของวันทำการก่อน (live/daily_check.py — ข้อ 3 รอบที่ 4):
  ขั้น 2 ดึงซ้ำ → ขั้น 3 สร้างจากแหล่งสำรองลง outputs/live_cache/ → ขั้น 4 วนดึงซ้ำจนเหลือ final_margin_min
  ก่อนเส้นตาย แล้วส่งด้วยข้อมูลล่าสุดที่มี · ทุกขั้นได้ไฟล์ส่ง และพิมพ์คำเตือนตัวใหญ่

→ <submission.dir>/<submission.filename>  (ไฟล์ส่งอาจารย์ — รูปแบบใน rules.yaml → submission)
→ <ไฟล์ส่ง>_detail.csv                      (คอลัมน์ตรวจสอบ ไม่ส่ง)

ร่างรอบเช้า (รอบที่ 5 ข้อ A5) — feature รอบเช้าใช้ข้อมูลถึงราคาปิดเมื่อวานเท่านั้น จึงคำนวณได้ตั้งแต่เย็นวันก่อน:
    python scripts/05_predict_live.py morning --draft      # หลัง 00_collect_daily.py (ทำให้อัตโนมัติแล้ว)
    → outputs/predictions/draft/<วันทำการถัดไป>_morning.csv (+ _detail.csv, .jsonl) — ไม่ลง log ส่งจริง
  เช้าวันจริง: รันปกติแล้วเทียบกับร่าง (ตรง → ส่งได้ทันที · ต่าง → ใช้ผลตอนเช้า + เหตุผลใน fallback_reason)
  รันตอนเช้าล้ม → ใช้ร่างเป็นไฟล์ส่ง พร้อม fallback_reason
→ outputs/logs/predictions.jsonl            (log ก่อนรู้ผล ห้ามแก้)
→ outputs/logs/shadow_log.jsonl             (รันเงา E0-ud + ราคาล่าสุด — ห้ามส่ง ใช้เทียบในรายงานเท่านั้น)
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import INTERVAL_MIN, OUTPUT_DIR, RAW_DIR, ROOT, load_rules, raw_source_dir, session_times
from forecast.data.io import drop_unfinished, merge_into_raw, raw_path
from forecast.live.daily_check import build_fallback_root, missing_daily, next_trading_day, prev_trading_day
from forecast.live.log_schema import PredictionLog, append_log, read_logs
from forecast.live.predict import predict_session

SESSION_TH = {"morning": "เช้า", "afternoon": "บ่าย"}

COLUMNS = {   # ชื่อภายในของคอลัมน์ไฟล์ส่ง (rules.yaml → submission.columns)
    "date": lambda r, p: r.date, "session": lambda r, p: r.session, "session_th": lambda r, p: SESSION_TH[r.session],
    "symbol": lambda r, p: r.symbol, "direction": lambda r, p: r.direction, "p_up": lambda r, p: round(r.p_up, 3),
    "q10": lambda r, p: p["q10"], "q25": lambda r, p: p["q25"], "q50": lambda r, p: p["q50"],
    "q75": lambda r, p: p["q75"], "q90": lambda r, p: p["q90"],
    "band50": lambda r, p: f"{p['q25']:.2f}–{p['q75']:.2f}", "band80": lambda r, p: f"{p['q10']:.2f}–{p['q90']:.2f}",
}


def warn(lines):
    bar = "!" * 78
    print(f"\n{bar}\n" + "\n".join(f"!!  {x}" for x in lines) + f"\n{bar}\n", flush=True)


def fetch_today(rules, today=True):
    tz = rules["timezone"]
    now = lambda: pd.Timestamp.now(tz=tz).tz_localize(None)
    iv = rules["data"]["interval"]
    jobs = {f"yfinance {iv}": lambda sym: ("yfinance", iv, _yf(sym, iv, "5d", rules)),
            "settrade 1d": lambda sym: ("settrade", "1d", _settrade_latest(sym, "1d", rules)),   # P(c) รอบเช้า
            "yfinance 1d": lambda sym: ("yfinance", "1d", _yf(sym, "1d", "5d", rules))}          # สำรองของ Settrade 1d
    for src in rules["live"]["sources"] if today else []:
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


def fetch_prev_day(rules, day):
    """ขั้น 2: ดึงข้อมูลรายวัน + 60m ซ้ำ และ Settrade 1m ของวันทำการก่อน (ใช้สร้างแท่งสำรองในขั้น 3)"""
    fetch_today(rules, today=False)
    for sym in rules["symbols"]:
        try:
            from forecast.data.settrade import fetch_day
            df = fetch_day(_settrade(), sym, day, rules)
            added, _ = merge_into_raw(raw_path(sym, rules, raw_source_dir(rules, "settrade"), "1m"), df)
            print(f"  ดึง settrade 1m {sym} {day:%d/%m}: +{added} แท่ง")
        except BaseException as e:
            print(f"  ดึง settrade 1m {sym} {day:%d/%m}: ล้ม ({type(e).__name__}: {str(e)[:80]})")


def ensure_daily(rules, args, date, session):
    """ข้อ 3 → (data_root หรือ None = data/raw, {หุ้น: [เหตุผล]})"""
    cfg = rules["live"]["daily_check"]
    src_root = args.data_root or RAW_DIR
    day = prev_trading_day(date)
    missing = missing_daily(rules, day, src_root)
    if not missing:
        print(f"ข้อมูลของวันทำการก่อน ({day:%d/%m}) ครบ: settrade 1d + yfinance 60m")
        return args.data_root, {}
    warn([f"ขั้น 1: ข้อมูลของวันทำการก่อน ({day:%d/%m}) ยังไม่มี — " +
          "; ".join(f"{s}: {', '.join(m)}" for s, m in missing.items())])
    fetching = not args.no_fetch and args.data_root is None
    for i in range(cfg["refetch_attempts"] if fetching else 0):
        print(f"ขั้น 2: ดึงซ้ำครั้งที่ {i + 1}")
        fetch_prev_day(rules, day)
        missing = missing_daily(rules, day, src_root)
        if not missing:
            warn([f"ขั้น 2: ดึงซ้ำแล้วได้ข้อมูลของ {day:%d/%m} ครบ"])
            return args.data_root, {}
        time.sleep(cfg["refetch_wait_sec"])
    out_root = OUTPUT_DIR / "live_cache" / f"{date:%Y-%m-%d}_{session}"
    give_up = session_times(rules, session, date)["deadline"] - pd.Timedelta(minutes=cfg["final_margin_min"])
    while True:
        notes = build_fallback_root(rules, missing, day, out_root, src_root)
        left = missing_daily(rules, day, out_root)
        if not left:
            warn(["ขั้น 3: สร้างแท่งของวันทำการก่อนจากแหล่งสำรอง (ไม่แก้ data/raw)"] +
                 [f"{s}: {n}" for s, ns in notes.items() for n in ns] + [f"สำเนาข้อมูล: {out_root}"])
            return out_root, {s: [f"ข้อมูลรายวันขั้น 3: {n}" for n in ns] for s, ns in notes.items()}
        if not fetching or pd.Timestamp.now(tz=rules["timezone"]) >= give_up:
            break
        print(f"ขั้น 4: ยังไม่ครบ ({left}) — ดึงซ้ำอีกครั้ง (รอได้ถึง {give_up:%H:%M})")
        time.sleep(cfg["refetch_wait_sec"])
        fetch_prev_day(rules, day)
        missing = missing_daily(rules, day, src_root)
        if not missing:
            warn([f"ขั้น 4: ดึงซ้ำแล้วได้ข้อมูลของ {day:%d/%m} ครบ"])
            return args.data_root, {}
    warn([f"ขั้น 4: ยังไม่มีข้อมูลของ {day:%d/%m} → ทำนายด้วย fd_lr บนข้อมูลล่าสุดที่มี — ตรวจ fallback_reason"] +
         [f"{s}: {n}" for s, ns in notes.items() for n in ns])
    reasons = {s: [f"ข้อมูลรายวันขั้น 4: ไม่มี {', '.join(m)} ของ {day:%d/%m} → ใช้ข้อมูลล่าสุดที่มี"]
               for s, m in left.items()}
    for s, ns in notes.items():   # แถวที่สร้างได้บางส่วนยังใช้ (เช่น 60m ได้ แต่ 1d ไม่ได้)
        reasons.setdefault(s, []).extend(f"ข้อมูลรายวันขั้น 3: {n}" for n in ns if "ไม่ได้" not in n)
    return out_root, reasons


def draft_path(date):
    return OUTPUT_DIR / "predictions" / "draft" / f"{date:%Y-%m-%d}_morning.jsonl"


def same_as_draft(r, d):
    """ทิศทาง P(ขึ้น) และราคาทุก quantile ตรงกับร่าง → (ตรงไหม, รายการที่ต่าง)"""
    diff = []
    if r.direction != d.direction or abs(r.p_up - d.p_up) > 1e-9:
        diff.append(f"P(ขึ้น) {d.p_up:.4f} → {r.p_up:.4f}")
    for k, v in r.price_quantiles.items():
        if abs(v - d.price_quantiles[k]) > 1e-9:
            diff.append(f"{k} {d.price_quantiles[k]:g} → {v:g}")
    return not diff, diff


def compare_draft(records, date):
    path = draft_path(date)
    if not path.exists():
        print(f"ไม่มีร่างเย็นวันก่อน ({path.name})")
        return
    drafts = {d["symbol"]: PredictionLog(**d) for d in read_logs(path)}
    lines = []
    for r in records:
        d = drafts.get(r.symbol)
        if d is None:
            continue
        ok, diff = same_as_draft(r, d)
        lines.append(f"{r.symbol}: " + ("ตรงกับร่าง" if ok else "ต่างจากร่าง — " + ", ".join(diff)))
        if not ok:
            r.fallback_reason = "; ".join(filter(None, [r.fallback_reason, "ต่างจากร่างเย็นวันก่อน (ใช้ผลตอนเช้า): " +
                                                         ", ".join(diff)]))
    print("เทียบร่างเย็นวันก่อน: " + " · ".join(lines))


def from_draft(date, err):
    """รันตอนเช้าล้ม → ใช้ร่างเย็นวันก่อนเป็นผลส่ง"""
    path = draft_path(date)
    if not path.exists():
        raise SystemExit(f"รันล้ม ({err}) และไม่มีร่างเย็นวันก่อน — ใช้แผนสำรองในคู่มือ (กรอกมือ)")
    out = []
    for d in read_logs(path):
        r = PredictionLog(**d)
        r.generated_at = str(datetime.now())
        r.fallback_reason = "; ".join(filter(None, [r.fallback_reason, f"ใช้ร่างเย็นวันก่อน (รันตอนเช้าล้ม: {err})"]))
        out.append(r)
    warn(["รันตอนเช้าล้ม → ใช้ร่างเย็นวันก่อนเป็นไฟล์ส่ง", err])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", choices=["morning", "afternoon"])
    ap.add_argument("--date", help="default = วันนี้ (--draft: วันทำการถัดไป)")
    ap.add_argument("--now", help="จำลองเวลาปัจจุบัน (ทดสอบย้อนหลัง)")
    ap.add_argument("--data-root", type=Path, help="โฟลเดอร์แทน data/raw (มี settrade/ yfinance/) — ใช้ซ้อม")
    ap.add_argument("--no-fetch", action="store_true", help="ไม่ดึงข้อมูลใหม่ ใช้ไฟล์ที่มีอยู่")
    ap.add_argument("--draft", action="store_true", help="ร่างรอบเช้าของวันทำการถัดไป (รันตอนเย็นหลังเก็บข้อมูล)")
    args = ap.parse_args()
    rules = load_rules()
    t0 = time.perf_counter()
    if not args.draft:
        args.date = args.date or datetime.now().strftime("%Y-%m-%d")
    if args.draft:
        if args.session != "morning":
            raise SystemExit("--draft ใช้กับรอบเช้าเท่านั้น")
        args.date = args.date or f"{next_trading_day(datetime.now()):%Y-%m-%d}"
        args.no_fetch = True
        args.now = args.now or f"{args.date} {rules['sessions']['morning']['cutoff']}"
    if not args.no_fetch and args.data_root is None:
        fetch_today(rules)
    date = pd.Timestamp(args.date)
    data_root, extra = ensure_daily(rules, args, date, args.session)

    try:
        records, shadows = predict_session(args.session, args.date, rules, now=args.now, data_root=data_root,
                                           extra_reasons=extra)
    except Exception as e:
        if args.session != "morning" or args.draft:
            raise
        records, shadows = from_draft(date, f"{type(e).__name__}: {str(e)[:120]}"), []
    else:
        if args.session == "morning" and not args.draft:
            compare_draft(records, date)
    # จำลองเวลา (--now) = ทดสอบย้อนหลัง/ซ้อม → log แยก ไม่ปนกับ log ส่งจริง · ร่างไม่ลง log
    kind = "draft" if args.draft else "dryrun" if args.now else ""
    logdir = OUTPUT_DIR / "logs" / kind
    if args.draft:
        draft_path(date).unlink(missing_ok=True)
    for sh in shadows if not args.draft else []:
        append_log(logdir / "shadow_log.jsonl", sh)
    sub = rules["submission"]
    rows, detail = [], []
    for r in records:
        if args.draft:
            append_log(draft_path(date), r)
        else:
            r.submitted_at = str(datetime.now())
            append_log(logdir / "predictions.jsonl", r)
        p = r.price_quantiles
        rows.append({name: COLUMNS[key](r, p) for name, key in sub["columns"].items()})
        detail.append({
            "วันที่": r.date, "รอบ": SESSION_TH[r.session], "หุ้น": r.symbol,
            "ทิศทาง": r.direction, "P(ขึ้น)": round(r.p_up, 3),
            "ราคาคาด": p["q50"], "ช่วง 50%": f"{p['q25']:.2f}–{p['q75']:.2f}",
            "ช่วง 80%": f"{p['q10']:.2f}–{p['q90']:.2f}",
            "ราคาอ้างอิง P(c)": r.reference_price, "ที่มา P(c)": r.reference_source,
            "P(ราคาเท่า P(c))": None if r.p_eq is None else round(r.p_eq, 3),
            "P(สูงกว่า)": None if r.p_higher is None else round(r.p_higher, 3),
            "P(ต่ำกว่า)": None if r.p_lower is None else round(r.p_lower, 3),
            "แหล่งข้อมูลวันนี้": r.data_source, "อายุข้อมูล (นาที)": None if r.data_age_min is None else round(r.data_age_min),
            "ราคาปัดช่อง (กฎ)": r.point_rule,
            "p ชั้น 1": None if r.p_stage1 is None else round(r.p_stage1, 4),
            "p ชั้น 2 (E13)": None if r.p_final is None else round(r.p_final, 4),
            "ชั้น 2 หมายเหตุ": r.stage2_reason or "",
            "fallback": r.fallback_reason or "",
        })

    table, detail = pd.DataFrame(rows), pd.DataFrame(detail)
    name = sub["filename"].format(date=args.date, session=args.session, session_th=SESSION_TH[args.session])
    out = ROOT / sub["dir"] / kind / name
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    if sub.get("detail"):
        detail.to_csv(out.with_name(out.stem + "_detail.csv"), index=False, encoding="utf-8-sig")
    print(detail.to_string(index=False))
    print("\nไฟล์ส่ง:")
    print(table.to_string(index=False))
    print(f"\n→ {out}")
    run_sec = time.perf_counter() - t0
    log = OUTPUT_DIR / "logs" / "run_time.csv"
    pd.DataFrame([{"session": args.session, "date": args.date, "started": datetime.now(), "run_sec": run_sec,
                   "fetched": not args.no_fetch}]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"ใช้เวลา {run_sec:.1f} วินาที (ดึงข้อมูล + ทำนาย) → {log}")
    deadline = session_times(rules, args.session, date)["deadline"]
    done = (pd.Timestamp(args.now).tz_localize(rules["timezone"]) + pd.Timedelta(seconds=run_sec) if args.now
            else pd.Timestamp.now(tz=rules["timezone"]))
    if args.draft:
        print(f"ร่างรอบเช้าของ {args.date} → {draft_path(date)} (ไม่ลง log ส่งจริง)")
        return
    print(f"ได้ไฟล์ส่ง {done:%H:%M:%S} — ก่อนเส้นตาย {deadline:%H:%M} อยู่ {(deadline - done).total_seconds() / 60:.1f} นาที"
          + (" (เวลาจำลอง + เวลารันจริง)" if args.now else ""))
    if detail["fallback"].astype(bool).any():
        warn(["มีรายการที่ใช้ทางสำรอง — ตรวจทุกบรรทัดในคอลัมน์ fallback ก่อนส่ง"] +
             [f"{x['หุ้น']}: {x['fallback']}" for _, x in detail.iterrows() if x["fallback"]])
    print(f"รันเงา {len(shadows)} รายการ → {logdir / 'shadow_log.jsonl'} (ไม่ส่ง)")


if __name__ == "__main__":
    main()
