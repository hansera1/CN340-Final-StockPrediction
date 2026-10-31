"""รายงานรอบที่ 4 (HTML → PDF) — ตัวเลขในตารางอ่านจาก results/round4/ ทั้งหมด

    python experiments/20_make_report_round4.py [--out outputs/reports/report_round4.pdf]

ต้องรันก่อน: experiments/17_label_variants.py pcoff / atc, 18_score_round4.py t4 / atc, 19_t5_xd.py
"""
import argparse
import importlib
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT

r3 = importlib.import_module("16_make_report_round3")
r2, r1 = r3.r2, r3.r1
esc, table, pct, ci_pp, forest_svg = r1.esc, r1.table, r1.pct, r2.ci_pp, r2.forest_svg
RES = ROOT / "results" / "round4"
GROUP_TH = {**r1.GROUP_TH, "ALL/morning": "รวมรอบเช้า", "ALL/afternoon": "รวมรอบบ่าย",
            "KBANK/ALL": "KBANK ทั้งวัน", "ADVANC/ALL": "ADVANC ทั้งวัน"}
ORDER = ["ALL", "ALL/morning", "ALL/afternoon", "KBANK/morning", "KBANK/afternoon", "ADVANC/morning", "ADVANC/afternoon"]


def baht_ci(row, key="d_mae_baht"):
    lo, hi = row["d_lo"], row["d_hi"]
    sig = " ✱" if lo > 0 or hi < 0 else ""
    return f"{row[key]:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


def price_table(t):
    rows = []
    for lab, g in t.groupby("label", sort=False):
        g = g.set_index("group")
        for k in ORDER:
            r = g.loc[k]
            rows.append([esc(lab) if k == "ALL" else "", GROUP_TH[k], int(r["n"]),
                         f'{int(r["n_shift_up"])} / {int(r["n_shift_down"])}', f'{r["mae_rule_baht"]:.3f}',
                         f'{r["mae_pc_baht"]:.3f}', baht_ci(r), pct(r["cover50"]), pct(r["cover80"])])
    return table(["โมเดลราคา", "ชุด", "รอบ", "เลื่อน ขึ้น / ลง", "MAE กฎ (บาท)", "MAE P(c) (บาท)",
                  "Δ MAE [95% CI]", "ครอบ 50%", "ครอบ 80%"], rows, "small")


def dryrun_table():
    d = pd.read_csv(RES / "dryrun_2026-10-30.csv")
    rows = [[r["รอบ"], r["หุ้น"], r["ทิศทาง"], f'{r["P(ขึ้น)"]:.3f}', f'{r["ราคาคาด"]:.2f}', esc(r["ช่วง 50%"]),
             esc(r["ช่วง 80%"]), f'{r["ราคาอ้างอิง P(c)"]:.2f}', esc(r["ที่มา P(c)"]), esc(r["แหล่งข้อมูลวันนี้"]),
             "–" if pd.isna(r["fallback"]) else esc(r["fallback"])] for _, r in d.iterrows()]
    return table(["รอบ", "หุ้น", "ทิศทาง", "P(ขึ้น)", "ราคาคาด", "ช่วง 50%", "ช่วง 80%", "P(c)", "ที่มา P(c)",
                  "แหล่งข้อมูลวันนี้", "fallback"], rows, "small")


def synth_table():
    d = pd.read_csv(RES / "t3_synth_check.csv")
    rows = [[r.date, r.symbol, esc(r.source), "ตรง" if r.ohlc_1d_match else "ไม่ตรง", f"{r.volume_ratio:.3f}",
             f"{r.bars_60m_match}/{r.bars_60m}"] for r in d.itertuples()]
    return table(["วัน", "หุ้น", "แหล่งที่ใช้สร้าง", "OHLC รายวัน เทียบ Settrade 1d", "volume ÷ ของจริง",
                  "แท่ง 60m ตรง yfinance"], rows, "small")


def atc_section():
    t = pd.read_csv(RES / "atc_direction.csv")
    m, b = t[t.model.str.startswith("fd_lr")].set_index("group"), t[t.model.str.startswith("E0")].set_index("group")
    rows = []
    for k in ORDER:
        x = m.loc[k].to_dict()
        rows.append([GROUP_TH[k], int(x["n_nonflat"]), pct(x["flat_rate"]), pct(x["acc_nf"]), pct(b.loc[k, "acc_nf"]),
                     pct(x["acc_fw"]), ci_pp(x, "d_vs_e0ud")])
    main = table(["ชุด", "รอบไม่ flat", "สัดส่วน flat", "fd_lr แม่น (ไม่นับ flat)", "E0-ud", "fd_lr (flat = ผิด)",
                  "Δ เทียบ E0-ud [95% CI]"], rows, "small")
    forest = forest_svg([(GROUP_TH[k], m.loc[k, "d_vs_e0ud"], m.loc[k, "d_lo"], m.loc[k, "d_hi"]) for k in ORDER])
    price = pd.read_csv(RES / "atc_price.csv")
    return main, forest, price_table(price)


def t5_table():
    t = pd.read_csv(RES / "t5_xd.csv")
    a = t[t.test.str.startswith("MAE")]
    rows = [[r.symbol, int(r.n_xd), f"{r.xd_mae_pc:.2f}", f"{r.xd_mae_pc_minus_div:.2f}", f"{r.other_mae_pc:.2f}"]
            for r in a.itertuples()]
    t1 = table(["หุ้น", "เช้าวัน XD", "MAE ราคาจุด = P(c) (บาท)", "MAE P(c) − ปันผล", "MAE วันปกติ"], rows, "small")
    b = t[t.test.str.startswith("fd_lr")]
    rows = [[r.symbol, int(r.n_xd), f"{pct(r.acc_after_xd)} [{pct(r.acc_after_lo)}, {pct(r.acc_after_hi)}]",
             int(r.n_other), pct(r.acc_other)] for r in b.itertuples()]
    t2 = table(["หุ้น", "รอบไม่ flat ใน 10 วันหลัง XD", "แม่น [95% CI]", "รอบไม่ flat วันอื่น", "แม่น วันอื่น"],
               rows, "small")
    days = a.loc[a.symbol != "ALL", ["symbol", "xd_days"]]
    return t1, t2, "; ".join(f"{r.symbol}: {r.xd_days}" for r in days.itertuples())


def build_html(nar):
    atc_main, atc_forest, atc_price = atc_section()
    t5a, t5b, xd_days = t5_table()
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><title>รายงานรอบที่ 4</title>
<style>{r2.CSS}</style></head><body>
<h1>ผลการทดสอบรอบที่ 4: แก้ระบบวันจริงและเตรียมส่งจริง 9–13 พ.ย.</h1>
<div class="sub">KBANK / ADVANC · walk-forward 24/04/2024 – 12/10/2026 (dataset ตัดที่ 29/10/2026 เท่ารอบที่ 3, ไม่แตะ holdout)
· สร้างเมื่อ {datetime.now():%d/%m/%Y %H:%M}</div>
{nar["summary"]}

<h2>1. แหล่งข้อมูลหลักและบั๊กที่พบ (ข้อ 0)</h2>
{nar["source"]}
<h3>1.1 Dry-run 30 ต.ค. ด้วยระบบสุดท้าย</h3>
{dryrun_table()}
{nar["dryrun"]}

<h2 class="pb">2. P(c) รอบเช้าของโมเดลราคา = ราคาปิดทางการ (ข้อ 4, T4)</h2>
{nar["t4_method"]}
{price_table(pd.read_csv(RES / "t4_price_official_pref.csv"))}
<p class="note">Δ MAE = MAE กฎค่ามัธยฐาน − MAE ของราคาจุด = P(c) แบบใหม่ (บวก = กฎแย่กว่า) · ✱ = CI ไม่ครอบ 0 ·
ช่วงราคาคิดแบบวันจริง (ราคาจุด × exp(z<sub>q</sub> − z<sub>50</sub>) ปัดช่องราคา)</p>
{nar["t4_result"]}

<h2 class="pb">3. ข้อมูลรายวันไม่สด: fallback แบบไล่ระดับ (ข้อ 3, T3)</h2>
{nar["t3_method"]}
<h3>3.1 แท่งที่สร้างจากแหล่งสำรอง เทียบของจริง</h3>
{synth_table()}
<h3>3.2 T3: dry-run บนสำเนาข้อมูลที่ลบแถวของเมื่อวาน</h3>
{nar["t3_result"]}

<h2 class="pb">4. เตรียมรองรับคำตอบอาจารย์ (ข้อ 1)</h2>
{nar["switches"]}
<h3>4.1 ถ้า 16:30 = ราคาปิด ATC: ผล walk-forward (ข้อ 1ค — เก็บไว้ ยังไม่ใช้)</h3>
{nar["atc_method"]}
{atc_main}
<figure><figcaption>ผลต่างความแม่น (ไม่นับ flat) ของ fd_lr เทียบ E0-ud บน label ATC และ 95% CI (จุด %)</figcaption>
{atc_forest}</figure>
<h3>4.2 กฎราคาบน label ATC</h3>
{atc_price}
{nar["atc_result"]}

<h2 class="pb">5. วันหยุดและวัน XD (ข้อ 5, T5)</h2>
{nar["calendar"]}
<h3>5.1 MAE รอบเช้าในวัน XD</h3>
{t5a}
<h3>5.2 ความแม่นของ fd_lr หลังวัน XD</h3>
{t5b}
<p class="note">วัน XD ในช่วง walk-forward — {esc(xd_days)}</p>
{nar["t5_result"]}

<h2>6. เอกสาร (ข้อ 7)</h2>
{nar["docs"]}

<h2 class="pb">7. งานที่เหลือก่อนส่งจริง</h2>
{nar["remaining"]}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "reports" / "report_round4.pdf")
    ap.add_argument("--narrative", type=Path, default=ROOT / "results" / "round4_narrative.py")
    args = ap.parse_args()
    ns = {}
    exec(args.narrative.read_text(encoding="utf-8"), ns)
    src = args.out.with_suffix(".html")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(build_html(ns["NARRATIVE"]), encoding="utf-8")
    browser = r1.find_browser()
    profile = Path(tempfile.gettempdir()) / "forecast-pdf-profile"
    args.out.unlink(missing_ok=True)
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                    f"--print-to-pdf={args.out.resolve()}", src.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    for _ in range(120):
        if args.out.exists() and args.out.stat().st_size > 0:
            size = args.out.stat().st_size
            time.sleep(1)
            if args.out.stat().st_size == size:
                break
        time.sleep(0.5)
    print("→", args.out)


if __name__ == "__main__":
    main()
