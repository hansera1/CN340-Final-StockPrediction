# ตั้ง VM (DigitalOcean) เป็นเครื่องหลัก — ทำทีละขั้น

VM: Ubuntu 24.04 (สิงคโปร์) · **VM = เครื่องหลัก, โน้ตบุ๊ก = เครื่องสำรอง** · ห้ามเปลี่ยนโมเดลที่ล็อก
`<VM_IP>` = IP ของ VM (ไม่ใส่ IP จริงใน repo เพราะ repo เป็นสาธารณะ)

> ห้ามคัดลอก `.env` ผ่านแชต อีเมล หรือ git — พิมพ์ใหม่บน VM ด้วย `nano` เท่านั้น

## 1. ระบบพื้นฐาน (เป็น root)

```bash
ssh root@<VM_IP>
curl -fsSL https://raw.githubusercontent.com/hansera1/CN340-Final-StockPrediction/main/scripts/setup_vm.sh -o setup_vm.sh
sudo bash setup_vm.sh system
```

ทำให้: เขตเวลา `Asia/Bangkok` + เปิด NTP (`timedatectl set-ntp true`), ติดตั้ง python3.12-venv / git / unzip / ufw / cron,
สร้าง user `forecast` (ระบบรันด้วย user นี้ ไม่ใช่ root) และคัดลอก SSH key ของ root ให้ `forecast`
ตรวจ: บรรทัด `Time zone: Asia/Bangkok (+07, +0700)` และ `System clock synchronized: yes`

## 2. ติดตั้งโปรเจกต์ (เป็น forecast)

```bash
sudo -iu forecast                    # หรือ ssh forecast@<VM_IP> จากเครื่องตัวเอง
curl -fsSL https://raw.githubusercontent.com/hansera1/CN340-Final-StockPrediction/main/scripts/setup_vm.sh -o ~/setup_vm.sh
bash ~/setup_vm.sh app
```

ทำให้: `git clone` → `~/kbank_advanc_forecast`, venv Python 3.12 (`.venv/`), `pip install -r requirements-live.txt`
(เฉพาะระบบวันจริง ไม่มี torch · เวอร์ชันตรึงให้ตรงเครื่องที่ล็อกโมเดล เพื่อให้เปิดไฟล์โมเดลได้)

## 3. สร้าง .env ด้วยมือ

```bash
cd ~/kbank_advanc_forecast
nano .env          # พิมพ์ตาม .env.example: SETTRADE_* 4 ค่า + NOTIFY_TELEGRAM_BOT_TOKEN + NOTIFY_TELEGRAM_CHAT_ID
chmod 600 .env
ls -l .env         # ต้องเป็น -rw------- forecast forecast
```

## 4. ตั้ง Telegram (บังคับ)

1. ใน Telegram เปิด **@BotFather** → `/newbot` → ตั้งชื่อ (เช่น `KBANK ADVANC Forecast`) และ username ที่ลงท้าย `bot`
   → ได้ **token** (`123456789:AA...`) — เก็บเป็นความลับ ห้ามโพสต์ในกลุ่ม/แชต
2. สร้างกลุ่ม Telegram ของทีม → เพิ่ม bot เข้ากลุ่ม → พิมพ์ข้อความในกลุ่ม 1 ครั้ง เช่น `/start@<username ของ bot>`
3. บน VM ใส่ token ใน `.env` (`NOTIFY_TELEGRAM_BOT_TOKEN=...`) แล้วหา chat id:
   ```bash
   .venv/bin/python scripts/12_telegram_setup.py chat-id     # แสดง chat id ของกลุ่ม (เลขติดลบ เช่น -100...)
   ```
4. ใส่ `NOTIFY_TELEGRAM_CHAT_ID=<เลขนั้น>` ใน `.env` แล้วทดสอบ:
   ```bash
   .venv/bin/python scripts/12_telegram_setup.py test        # กลุ่มต้องได้ข้อความ "[ชื่อเครื่อง] ℹ️ ทดสอบแจ้งเตือน..."
   ```
5. ทำข้อ 3–4 ซ้ำบนโน้ตบุ๊ก (ใช้ bot และกลุ่มเดียวกันได้ — ข้อความขึ้นต้นด้วยชื่อเครื่องเสมอ)

ข้อความ "ได้ไฟล์ส่ง" มีผลทำนายครบทั้ง 2 หุ้นทุกคอลัมน์ของไฟล์ส่ง และชื่อเครื่อง — ส่งต่อให้อาจารย์ได้ทันที

## 5. ข้อมูลและโมเดลจากโน้ตบุ๊ก

บนโน้ตบุ๊ก (PowerShell):
```powershell
python scripts/11_backup_bundle.py
scp outputs\backup\backup_full_<เครื่อง>_<เวลา>.zip forecast@<VM_IP>:~/
```
บน VM:
```bash
cd ~/kbank_advanc_forecast
.venv/bin/python scripts/11_backup_bundle.py --restore ~/backup_full_<เครื่อง>_<เวลา>.zip
.venv/bin/python scripts/00_collect_daily.py --force      # VM ดึงข้อมูลเอง + ร่างรอบเช้า + β ชั้น 2 ของเครื่องนี้
.venv/bin/python scripts/08_check_ready.py                # ต้องไม่มี FAIL
```
zip มีโมเดลที่ล็อก (รวม R4ATC), ข้อมูลดิบ KBANK/ADVANC และ log/ร่างของโน้ตบุ๊ก (ไปอยู่ `outputs/logs/from_<เครื่อง>/` ไม่ทับของ VM)
ไม่มี `.env` และไม่มี β ชั้น 2 (แต่ละเครื่องฝึกเองตอนเย็นจาก log ของตัวเอง)

## 6. ตั้งเวลา (cron)

```bash
bash scripts/install_cron.sh                  # งานเย็น 17:30 + watchdog 09:20 / 12:20 (จันทร์–ศุกร์)
bash scripts/install_cron.sh --with-predict   # หลังตั้งจุดตัดจริง: + รอบเช้า/บ่ายตามจุดตัด
crontab -l                                    # ตรวจ
```

- `CRON_TZ=Asia/Bangkok` และเขตเวลาระบบเป็น Asia/Bangkok (สคริปต์หยุดเองถ้าไม่ใช่ UTC+7)
- วันหยุดตาม `data/external/calendar.csv` สคริปต์ข้ามเอง · `flock` กันงานเดียวกันรันซ้อน
- log: `outputs/logs/cron_<งาน>.log` · แก้จุดตัดใน `rules.yaml` แล้วต้องรัน `install_cron.sh` ใหม่

## 7. ความปลอดภัย (เป็น root — หลังทดสอบว่า `ssh forecast@<VM_IP>` เข้าได้ด้วย key)

```bash
sudo bash setup_vm.sh harden
```
ทำให้: `ufw` เปิดเฉพาะ SSH · SSH ใช้ key เท่านั้น (`PasswordAuthentication no`, `PermitRootLogin prohibit-password`)
· สคริปต์ไม่ทำถ้ายังไม่มี SSH key (กันล็อกตัวเองออก) · **อย่าปิด session เดิมจนกว่าจะทดสอบ ssh จากอีก terminal ผ่าน**

## 8. วันที่ 2 พ.ย. — วัดความหน่วงบน VM (เครื่องที่ใช้ส่งจริง)

```bash
cd ~/kbank_advanc_forecast
# ~10:30 และ ~14:30 (ช่วงตลาดเปิด)
.venv/bin/python scripts/00_fetch_settrade.py --latency 30
.venv/bin/python scripts/00_fetch_yfinance.py --latency 30
# 12:15 — จับเวลารอบบ่ายจริงบน VM (ดึงข้อมูล + ทำนาย)
.venv/bin/python scripts/05_predict_live.py afternoon
# เสนอจุดตัดจากผลของ VM (latency.csv / run_time.csv บันทึกชื่อเครื่องทุกแถว)
.venv/bin/python scripts/07_propose_cutoff.py
```
ตั้งจุดตัดใน `rules.yaml` จากผลของ VM → `results/latency_2026-11-02.md` → `install_cron.sh --with-predict` (VM) และ `setup_tasks.ps1 -WithPredict` (โน้ตบุ๊ก)
แล้วทำตามแผนรอบที่ 6 ต่อ (ภาคผนวก E13 ต้อง commit ก่อนรัน backtest)

## 9. หลัง refit ศ. 6 พ.ย. — ซิงก์โมเดล

refit ทำที่โน้ตบุ๊ก (`python scripts/04_lock_model.py`) แล้ว:
```powershell
python scripts/11_backup_bundle.py --models-only
scp outputs\backup\backup_models_<เครื่อง>_<เวลา>.zip forecast@<VM_IP>:~/
```
บน VM:
```bash
.venv/bin/python scripts/11_backup_bundle.py --restore ~/backup_models_<เครื่อง>_<เวลา>.zip
.venv/bin/python scripts/08_check_ready.py     # trained_until ต้องเป็น 2026-11-06 ทุกโมเดล
```
และรัน `08_check_ready.py` บนโน้ตบุ๊กด้วย — ทั้งสองเครื่องต้องไม่มี FAIL

## 10. อัปเดตโค้ดบน VM (เมื่อ push โค้ดใหม่)

```bash
cd ~/kbank_advanc_forecast && git pull --ff-only && .venv/bin/python scripts/08_check_ready.py
```
หลังล็อกโค้ด 5 พ.ย. ห้าม pull โค้ดที่เปลี่ยนโมเดลระหว่าง 9–13 พ.ย.
