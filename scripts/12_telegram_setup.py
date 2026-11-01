"""ตั้งแจ้งเตือน Telegram (งาน 9 ข้อ 4 / VM ข้อ 3) — วิธีสร้าง bot: docs/vm_setup.md

    python scripts/12_telegram_setup.py chat-id    # อ่าน NOTIFY_TELEGRAM_BOT_TOKEN จาก .env → แสดง chat id ของกลุ่มที่ bot อยู่
    python scripts/12_telegram_setup.py test       # ส่งข้อความทดสอบ (ใช้ chat id ใน .env)

ไม่พิมพ์ token · chat id ไม่ใช่ความลับระดับเดียวกับ token แต่ให้ใส่ใน .env เท่านั้น ไม่ commit
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request

import _setup  # noqa: F401

from forecast.live.notify import notify
from forecast.runtime import read_env


def chat_ids(token):
    url = f"https://api.telegram.org/bot{urllib.parse.quote(token)}/getUpdates"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
    except Exception as e:   # ไม่พิมพ์ข้อความ error เต็ม (อาจมี URL ที่มี token)
        raise SystemExit(f"เรียก Telegram ไม่ได้ ({type(e).__name__}) — ตรวจ token ใน .env")
    found = {}
    for u in data.get("result", []):
        msg = u.get("message") or u.get("channel_post") or u.get("my_chat_member") or {}
        chat = msg.get("chat") or {}
        if "id" in chat:
            found[chat["id"]] = f"{chat.get('type')} · {chat.get('title') or chat.get('username') or chat.get('first_name')}"
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["chat-id", "test"])
    args = ap.parse_args()
    env = read_env(["NOTIFY_TELEGRAM_BOT_TOKEN", "NOTIFY_TELEGRAM_CHAT_ID"])
    if not env["NOTIFY_TELEGRAM_BOT_TOKEN"]:
        raise SystemExit("ยังไม่มี NOTIFY_TELEGRAM_BOT_TOKEN ใน .env")
    if args.what == "chat-id":
        found = chat_ids(env["NOTIFY_TELEGRAM_BOT_TOKEN"])
        if not found:
            print("ยังไม่เห็นข้อความใด — เพิ่ม bot เข้ากลุ่ม แล้วพิมพ์ข้อความในกลุ่ม (เช่น /start@<ชื่อbot>) แล้วรันใหม่")
            return 1
        print("chat id ที่ bot เห็น (กลุ่มมักเป็นเลขติดลบ):")
        for cid, desc in found.items():
            print(f"  {cid}   {desc}")
        print("→ ใส่ใน .env: NOTIFY_TELEGRAM_CHAT_ID=<เลขของกลุ่ม>")
        return 0
    if not env["NOTIFY_TELEGRAM_CHAT_ID"]:
        raise SystemExit("ยังไม่มี NOTIFY_TELEGRAM_CHAT_ID ใน .env — รัน chat-id ก่อน")
    res = notify("info", "ทดสอบแจ้งเตือนจากระบบทำนาย KBANK / ADVANC", ["ถ้าเห็นข้อความนี้ การตั้งค่า Telegram ใช้ได้"])
    return 0 if res.get("telegram") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
