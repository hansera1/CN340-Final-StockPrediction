"""แจ้งเตือนกลุ่ม (งาน 9 ข้อ 4) — ช่องทางตั้งใน .env (ตั้งได้หลายช่องพร้อมกัน, ไม่ตั้ง = เก็บลง outbox อย่างเดียว)

    NOTIFY_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
    NOTIFY_SLACK_WEBHOOK=https://hooks.slack.com/services/...
    NOTIFY_TELEGRAM_BOT_TOKEN=...   และ   NOTIFY_TELEGRAM_CHAT_ID=...

ทุกข้อความถูกเก็บใน outputs/logs/notify_outbox.jsonl (เวลา, ประเภท, ข้อความ, ผลการส่งต่อช่องทาง)
การส่งล้มทุกแบบถูกจับไว้ → ไม่กระทบการสร้างไฟล์ส่ง · ไม่พิมพ์/บันทึก URL หรือ token (บันทึกแค่ชนิด error / HTTP status)
"""
import json
import urllib.parse
import urllib.request
from datetime import datetime

from forecast.config import OUTPUT_DIR
from forecast.runtime import read_env

KEYS = ["NOTIFY_DISCORD_WEBHOOK", "NOTIFY_SLACK_WEBHOOK", "NOTIFY_TELEGRAM_BOT_TOKEN", "NOTIFY_TELEGRAM_CHAT_ID"]
ICON = {"ready": "✅", "fallback": "⚠️", "failed": "❌", "missing": "🚨", "evening": "🌙", "info": "ℹ️"}
OUTBOX = OUTPUT_DIR / "logs" / "notify_outbox.jsonl"


def _post(url, payload, timeout=8):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "User-Agent": "forecast-notify"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def channels(env=None):
    """→ {ชื่อช่องทาง: ฟังก์ชันส่ง(text)} ตามค่าที่ตั้งใน .env"""
    e = env if env is not None else read_env(KEYS)
    out = {}
    if e.get("NOTIFY_DISCORD_WEBHOOK"):
        out["discord"] = lambda t, u=e["NOTIFY_DISCORD_WEBHOOK"]: _post(u, {"content": t[:1900]})
    if e.get("NOTIFY_SLACK_WEBHOOK"):
        out["slack"] = lambda t, u=e["NOTIFY_SLACK_WEBHOOK"]: _post(u, {"text": t})
    if e.get("NOTIFY_TELEGRAM_BOT_TOKEN") and e.get("NOTIFY_TELEGRAM_CHAT_ID"):
        tok, chat = e["NOTIFY_TELEGRAM_BOT_TOKEN"], e["NOTIFY_TELEGRAM_CHAT_ID"]
        out["telegram"] = lambda t: _post(f"https://api.telegram.org/bot{urllib.parse.quote(tok)}/sendMessage",
                                          {"chat_id": chat, "text": t[:4000]})
    return out


def _safe_error(e):
    code = getattr(e, "code", None)
    return f"{type(e).__name__}" + (f" HTTP {code}" if code else "")


def notify(kind, title, lines=(), outbox=None, chans=None):
    """ส่งข้อความ → {ช่องทาง: "ok" / ชนิด error} · ไม่มีวัน raise"""
    try:
        text = f"{ICON.get(kind, '')} {title}".strip() + ("\n" + "\n".join(str(x) for x in lines) if lines else "")
        try:
            chans = channels() if chans is None else chans
        except Exception as e:
            chans, result = {}, {"config": _safe_error(e)}
        else:
            result = {}
        for name, send in chans.items():
            try:
                send(text)
                result[name] = "ok"
            except Exception as e:
                result[name] = _safe_error(e)
        path = outbox or OUTBOX
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "kind": kind, "text": text,
                                "sent": result or "ไม่ได้ตั้งช่องทาง (outbox อย่างเดียว)"}, ensure_ascii=False) + "\n")
        print(f"แจ้งเตือน [{kind}] → " + (", ".join(f"{k}: {v}" for k, v in result.items()) or "outbox (ยังไม่ตั้งช่องทางใน .env)"))
        return result
    except Exception:
        return {}
