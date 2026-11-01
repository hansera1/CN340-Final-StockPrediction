"""ตรวจเครื่องตอนเริ่มทุกสคริปต์ (งาน 9 ข้อ 1–2) — เรียกจาก scripts/_setup.py

- พิมพ์และบันทึก (outputs/logs/run_start.jsonl): เวลาปัจจุบัน + เขตเวลา, commit hash, ชื่อเครื่อง, เครือข่าย
- เขตเวลาเครื่องต้องเป็น Asia/Bangkok (UTC+7) — ไม่ใช่: ลองตั้ง TZ ภายในโปรแกรม (POSIX) ไม่ได้ → หยุดพร้อมข้อความ
- นาฬิกาเครื่องเทียบ Date header ของเว็บอ้างอิง ต่างเกิน 60 วินาที → เตือนตัวใหญ่ + บันทึก (ตรวจไม่ได้ → บันทึกว่าตรวจไม่ได้ ไม่หยุด)
ไม่อ่าน/ไม่พิมพ์ค่าใน .env
"""
import json
import os
import platform
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from forecast.config import OUTPUT_DIR, ROOT

BKK_OFFSET = timedelta(hours=7)
CLOCK_REFS = ["https://www.settrade.com", "https://www.set.or.th", "https://www.google.com"]
MAX_SKEW_SEC = 60


def local_offset():
    return datetime.now().astimezone().utcoffset()


def check_timezone():
    """→ ข้อความหมายเหตุ (None = ปกติ) · ตั้งไม่ได้ → SystemExit"""
    if local_offset() == BKK_OFFSET:
        return None
    if hasattr(time, "tzset"):
        os.environ["TZ"] = "Asia/Bangkok"
        time.tzset()
        if local_offset() == BKK_OFFSET:
            return "เขตเวลาเครื่องไม่ใช่ Asia/Bangkok — ตั้ง TZ=Asia/Bangkok ภายในโปรแกรมแล้ว"
    raise SystemExit(f"หยุด: เขตเวลาเครื่องคือ UTC{_fmt(local_offset())} ไม่ใช่ Asia/Bangkok (UTC+07:00) "
                     "— ตั้งเขตเวลาของ Windows เป็น (UTC+07:00) Bangkok, Hanoi, Jakarta แล้วรันใหม่")


def _fmt(off):
    s = int(off.total_seconds())
    return f"{'+' if s >= 0 else '-'}{abs(s) // 3600:02d}:{abs(s) % 3600 // 60:02d}"


def commit_hash():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                           timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (h or "unknown") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def network_name():
    """ชื่อ Wi-Fi (SSID) ถ้ามี ไม่งั้น 'wired/unknown'"""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in out.splitlines():
                k, _, v = line.partition(":")
                if k.strip() == "SSID":
                    return f"Wi-Fi {v.strip()}"
    except Exception:
        pass
    return "wired/unknown"


def public_network(name):
    """ชื่อเครือข่ายสำหรับไฟล์ที่ commit (repo สาธารณะ) — ซ่อนชื่อ Wi-Fi ซึ่งอาจบอกที่อยู่ เก็บชื่อเต็มใน log เครื่องเท่านั้น"""
    return "Wi-Fi" if str(name).startswith("Wi-Fi") else name


def reference_time(urls=CLOCK_REFS, timeout=4):
    """→ (เวลา UTC จาก Date header, url) หรือ (None, เหตุผล)"""
    for url in urls:
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "forecast-clock-check"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = r.headers.get("Date")
            if d:
                return parsedate_to_datetime(d).astimezone(timezone.utc), url
        except Exception as e:
            last = type(e).__name__
    return None, f"ตรวจนาฬิกาไม่ได้ ({last})"


def clock_skew(ref=None):
    """→ (นาฬิกาเครื่อง − เวลาอ้างอิง เป็นวินาที หรือ None, ที่มา)"""
    ref, src = ref or reference_time()
    if ref is None:
        return None, src
    return (datetime.now(timezone.utc) - ref).total_seconds(), src


def machine_info():
    return {"host": platform.node(), "network": network_name(), "commit": commit_hash(), "python": platform.python_version()}


def banner(lines):
    bar = "!" * 78
    return f"\n{bar}\n" + "\n".join(f"!!  {x}" for x in lines) + f"\n{bar}\n"


def startup(script, log=None, check_clock=True):
    """ตรวจเขตเวลา + นาฬิกา แล้วพิมพ์/บันทึก → record"""
    tz_note = check_timezone()
    now = datetime.now().astimezone()
    rec = {"script": script, "started_at": now.isoformat(timespec="seconds"), "utc_offset": _fmt(now.utcoffset()),
           "tz_note": tz_note, **machine_info(), "clock_skew_sec": None, "clock_ref": None, "clock_warning": False}
    if check_clock:
        skew, src = clock_skew()
        rec["clock_skew_sec"], rec["clock_ref"] = (None if skew is None else round(skew, 1)), src
        rec["clock_warning"] = skew is not None and abs(skew) > MAX_SKEW_SEC
    print(f"[{script}] {now:%Y-%m-%d %H:%M:%S} UTC{rec['utc_offset']} (Asia/Bangkok) · commit {rec['commit']} · "
          f"เครื่อง {rec['host']} · {rec['network']}"
          + (f" · นาฬิกาต่างจากอ้างอิง {rec['clock_skew_sec']:+.0f} วินาที" if rec["clock_skew_sec"] is not None
             else f" · {rec['clock_ref']}" if check_clock else ""), flush=True)
    if tz_note:
        print(tz_note)
    if rec["clock_warning"]:
        print(banner([f"นาฬิกาเครื่องต่างจากเวลาอ้างอิง {rec['clock_skew_sec']:+.0f} วินาที (เกิน {MAX_SKEW_SEC} วินาที)",
                      f"อ้างอิง: {rec['clock_ref']} — ตั้งเวลา Windows: Settings → Time → Sync now",
                      "จุดตัดและเส้นตายคิดจากนาฬิกาเครื่อง: ตรวจเวลาให้ถูกก่อนส่ง"]), flush=True)
    log = log or OUTPUT_DIR / "logs" / "run_start.jsonl"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return rec


def read_env(keys, env_file=ROOT / ".env"):
    """ค่าจาก .env (ก่อน) แล้วเติมจาก environment — คืนค่าให้โปรแกรมใช้เท่านั้น ห้ามพิมพ์/บันทึก"""
    vals = dict.fromkeys(keys)
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in vals and v:
                vals[k] = v
    return {k: vals[k] or os.environ.get(k) for k in keys}
