"""งาน 9 ข้อ 5: ตรวจความพร้อมของเครื่อง (ใช้ได้ทั้งเครื่องหลักและเครื่องสำรอง) — scripts/08_check_ready.py

แต่ละข้อ → (ชื่อ, PASS / WARN / FAIL, รายละเอียด) · ไม่พิมพ์ค่าใน .env (บอกแค่ว่ามี/ไม่มีชื่อไหน)
"""
import pickle
from datetime import datetime

import pandas as pd

from forecast.config import OUTPUT_DIR, RAW_DIR
from forecast.data.settrade import ENV_KEYS
from forecast.live.daily_check import missing_daily, next_trading_day, prev_trading_day, is_trading_day
from forecast.live.evening import draft_info, stage2_info
from forecast.live.notify import KEYS as NOTIFY_KEYS
from forecast.runtime import MAX_SKEW_SEC, check_timezone, clock_skew, read_env

LIVE_KEYS = ["direction", "price_eq", "price_up", "shadow", "flat_dir"]


def targets(rules, now=None):
    """วันที่ของรอบเช้าและรอบบ่ายถัดไปที่ต้องเตรียม (ณ เวลา now)"""
    now = pd.Timestamp(now or datetime.now())
    today = now.normalize()
    am_deadline = pd.Timestamp(f"{today:%Y-%m-%d} {rules['sessions']['morning']['submit_deadline']}")
    pm_deadline = pd.Timestamp(f"{today:%Y-%m-%d} {rules['sessions']['afternoon']['submit_deadline']}")
    trading = is_trading_day(today)
    am = today if trading and now < am_deadline else next_trading_day(today)
    pm = today if trading and now < pm_deadline else next_trading_day(today)
    return am, pm


def check_env():
    e = read_env(ENV_KEYS)
    miss = [k for k, v in e.items() if not v]
    n = read_env(NOTIFY_KEYS)
    chans = [c for c, ks in {"discord": ["NOTIFY_DISCORD_WEBHOOK"], "slack": ["NOTIFY_SLACK_WEBHOOK"],
                             "telegram": ["NOTIFY_TELEGRAM_BOT_TOKEN", "NOTIFY_TELEGRAM_CHAT_ID"]}.items()
             if all(n.get(k) for k in ks)]
    out = [(".env Settrade", "FAIL" if miss else "PASS", f"ขาด {', '.join(miss)}" if miss else "ครบ 4 ค่า")]
    out.append((".env แจ้งเตือน", "PASS" if chans else "WARN", ", ".join(chans) if chans else "ยังไม่ตั้งช่องทาง (เก็บลง outbox อย่างเดียว)"))
    return out


def check_models(rules):
    live = rules["live"]
    prefix = f"{live['model_dir']}/" if live.get("model_dir") else ""
    out = []
    for k in LIVE_KEYS:
        spec = prefix + live[k]
        trained, bad = set(), []
        for sym in rules["symbols"]:
            for sess in rules["sessions"]:
                p = OUTPUT_DIR / "models" / spec / f"{sym}_{sess}.pkl"
                try:
                    with open(p, "rb") as f:
                        trained.add(pickle.load(f)["trained_until"])
                except Exception as e:
                    bad.append(f"{sym}_{sess} ({type(e).__name__})")
        out.append((f"โมเดล {k} ({spec})", "FAIL" if bad else "PASS",
                    f"โหลดไม่ได้: {', '.join(bad)}" if bad else f"4 ไฟล์ trained_until {', '.join(sorted(trained))}"))
    return out


def check_data(rules, am):
    day = prev_trading_day(am)
    miss = missing_daily(rules, day, RAW_DIR)
    return [(f"ข้อมูลถึงวันทำการก่อน ({day:%d/%m})", "FAIL" if miss else "PASS",
             "; ".join(f"{s}: {', '.join(m)}" for s, m in miss.items()) if miss else "settrade 1d + yfinance 60m ครบ")]


def check_prepared(am, pm):
    has_draft, mv = draft_info(am)
    has_beta, reason = stage2_info(pm)
    return [(f"ร่างรอบเช้า {am:%d/%m}", "PASS" if has_draft else "FAIL", mv or "ไม่มีไฟล์ร่าง (รัน 00_collect_daily.py)"),
            (f"β ชั้น 2 รอบบ่าย {pm:%d/%m}", "PASS" if has_beta else "WARN",
             (reason or "พร้อม") if has_beta else "ไม่มีไฟล์ (E13 เป็นรันเงา — ไม่กระทบไฟล์ส่ง)")]


def check_clock_tz():
    try:
        note = check_timezone()
        tz = ("เขตเวลา", "PASS", note or "Asia/Bangkok (UTC+07:00)")
    except SystemExit as e:
        tz = ("เขตเวลา", "FAIL", str(e))
    skew, src = clock_skew()
    if skew is None:
        ck = ("นาฬิกา", "WARN", src)
    else:
        ck = ("นาฬิกา", "FAIL" if abs(skew) > MAX_SKEW_SEC else "PASS", f"ต่างจาก {src} {skew:+.0f} วินาที")
    return [tz, ck]


def check_settrade(rules):
    try:
        from forecast.data.settrade import connect, fetch_latest
        d = fetch_latest(connect(), rules["symbols"][0], "1d", rules, limit=1)
        return [("Settrade API", "PASS", f"{rules['symbols'][0]} 1d ล่าสุด {d['timestamp'].max():%d/%m/%Y}")]
    except BaseException as e:   # รวม SystemExit จาก .env ไม่ครบ
        return [("Settrade API", "FAIL", type(e).__name__)]


def run_checks(rules, now=None, settrade=True):
    am, pm = targets(rules, now)
    rows = check_clock_tz() + check_env() + check_models(rules) + check_data(rules, am) + check_prepared(am, pm)
    if settrade:
        rows += check_settrade(rules)
    return pd.DataFrame(rows, columns=["check", "status", "detail"]), am, pm
