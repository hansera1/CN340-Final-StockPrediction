"""รายงานรอบที่ 5 (HTML → PDF) — ตัวเลขในตารางอ่านจาก results/round5/ และ results/e13/

    python experiments/23_make_report_round5.py [--out outputs/reports/report_round5.pdf]

ต้องรันก่อน: experiments/22_b_decimal_price.py, 21_e13_stage2.py e2, scripts/01b_validate_sources.py (คัดลอกผลลง results/round5/)
"""
import argparse
import importlib
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT

r4 = importlib.import_module("20_make_report_round4")
r2, r1 = r4.r2, r4.r1
esc, table, pct = r1.esc, r1.table, r1.pct
RES = ROOT / "results" / "round5"
E13 = ROOT / "results" / "e13"
GROUP_TH = r4.GROUP_TH
SESS_TH = {"morning": "เช้า", "afternoon": "บ่าย"}


def ci(m, lo, hi, d=3):
    sig = " ✱" if lo > 0 or hi < 0 else ""
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{sig}"


def b_tables():
    t = pd.read_csv(RES / "b_decimal_price.csv").set_index("group")
    rows = [[GROUP_TH[k], int(r["n"]), f'{r["mae_decimal"]:.3f}', f'{r["mae_rule"]:.3f}', f'{r["mae_pc"]:.3f}',
             f'{r["mean_abs_shift_baht"]:.3f}', ci(r["d_vs_rule"], r["d_vs_rule_lo"], r["d_vs_rule_hi"]),
             ci(r["d_vs_pc"], r["d_vs_pc_lo"], r["d_vs_pc_hi"])] for k, r in t.loc[r4.ORDER].iterrows()]
    price = table(["ชุด", "รอบ", "MAE ไม่ปัดช่อง", "MAE กฎค่ามัธยฐาน", "MAE P(c)", "|ราคา − P(c)| เฉลี่ย",
                   "Δ เทียบกฎ [95% CI]", "Δ เทียบ P(c) [95% CI]"], rows, "small")
    s = pd.read_csv(RES / "b_direction_share.csv")
    rows = [[r.symbol, SESS_TH[r.session], r.n_pred, pct(r.share_pred_down), pct(r.share_down_nonflat_test),
             pct(r.share_down_nonflat_first_train)] for r in s.itertuples()]
    share = table(["หุ้น", "รอบ", "รอบทดสอบ", "fd_lr ทาย \"ลง\"", "วันไม่ flat ที่ลงจริง (ช่วงทดสอบ)",
                   "วันไม่ flat ที่ลง (ชุดฝึกแรก)"], rows, "small")
    return price, share


def t6_table():
    d = pd.read_csv(RES / "validate_label_points.csv")
    rows = []
    for p, g in d.groupby("point", sort=False):
        ok = (g["yf60m"] == g["settrade_1m"]).sum()
        rows.append([p, len(g), f"{ok}/{len(g)}"])
    return table(["ราคาจุด label", "จุดที่เทียบ (วัน × หุ้น)", "yfinance 60m ตรง Settrade 1m"], rows, "small")


def e2_tables():
    s = pd.read_csv(E13 / "e2_summary.csv").iloc[0]
    summ = table(["จุดตัด", "วัน", "แท่งภาคเช้าที่เทียบ", "แท่งต่าง ≤ 1 ช่อง", "ต่างมากสุด (ช่อง)", "เครื่องหมาย f5 ตรงทุกวัน", "ผ่าน"],
                 [[s["cutoff"], int(s["n_days"]), int(s["n_bars"]), pct(s["share_bars_within_1_tick"]),
                   f'{s["max_diff_ticks"]:.0f}', "ใช่" if s["f5_signs_all_match"] else "ไม่", "ผ่าน" if s["pass"] else "ไม่ผ่าน"]],
                 "small")
    f = pd.read_csv(E13 / "e2_f5_compare.csv")
    rows = [[r.date, r.symbol, f"{r.f5_r_last15_st:+.4f} / {r.f5_r_last15_yf:+.4f}",
             f"{r.f5_close_vs_twap_st:+.4f} / {r.f5_close_vs_twap_yf:+.4f}",
             f"{r.f5_range_pos_st:.3f} / {r.f5_range_pos_yf:.3f}"] for r in f.itertuples()]
    detail = table(["วัน", "หุ้น", "f5_r_last15 (Settrade / yf)", "f5_close_vs_twap", "f5_range_pos"], rows, "small")
    return summ, detail


def build_html(nar):
    b_price, b_share = b_tables()
    e2_summ, e2_detail = e2_tables()
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><title>รายงานรอบที่ 5</title>
<style>{r2.CSS}</style></head><body>
<h1>ผลการทดสอบรอบที่ 5: ปิดจุดเสี่ยงของระบบวันจริง และเตรียม E13 แบบรันเงา</h1>
<div class="sub">KBANK / ADVANC · ทำวันที่ 1 พ.ย. 2026 (ฉบับก่อนวัดความหน่วง — ผล A3 และ E13 backtest จะเพิ่มหลัง 2 พ.ย.)
· สร้างเมื่อ {datetime.now():%d/%m/%Y %H:%M}</div>
{nar["summary"]}

<h2>1. สิ่งที่ทำในรอบนี้</h2>
{nar["done"]}

<h2 class="pb">2. Fallback ทุกขั้นเป็นโมเดล ML (ข้อ A2)</h2>
{nar["a2"]}

<h2>3. ร่างรอบเช้าตั้งแต่เย็นวันก่อน (ข้อ A5) และจุดตัด (ข้อ A3)</h2>
{nar["a5"]}

<h2 class="pb">4. ราคาที่ส่ง: ตัวเลือกไม่ปัดช่อง (ข้อ B)</h2>
{nar["b_method"]}
{b_price}
<p class="note">Δ = MAE ไม่ปัดช่อง − MAE ตัวเทียบ (บวก = แย่กว่า) · ✱ = CI ไม่ครอบ 0 · block bootstrap รายวัน 2,000 ครั้ง ·
walk-forward 24/04/2024 – 12/10/2026 ไม่แตะ holdout</p>
{nar["b_result"]}
<h3>4.1 ทำไมราคาจากโมเดลมักเท่าราคาล่าสุด และทำไม ADVANC บ่ายทาย "ลง" บ่อย</h3>
{b_share}
{nar["b_share"]}

<h2 class="pb">5. ตรวจงานค้างจากรอบที่ 4 (ข้อ C) และ T6</h2>
{nar["c"]}
<h3>5.1 T6: ราคา label จากแท่ง 60m เทียบ Settrade 1m (26–30 ต.ค.)</h3>
{t6_table()}
{nar["t6"]}

<h2 class="pb">6. E13: ชั้น 2 ปรับทิศทางรอบบ่ายด้วยแท่ง 5m (การทดลอง)</h2>
{nar["e13"]}
<h3>6.1 E-2 ตรวจแหล่งข้อมูล (ดูล่วงหน้าที่จุดตัดชั่วคราว)</h3>
{e2_summ}
{e2_detail}
{nar["e2"]}

<h2 class="pb">7. คู่มือ ปฏิทิน และสวิตช์ตามคำตอบอาจารย์ (ข้อ A1, A4, D)</h2>
{nar["a1"]}

<h2>8. งานที่เหลือ</h2>
{nar["remaining"]}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "reports" / "report_round5.pdf")
    ap.add_argument("--narrative", type=Path, default=ROOT / "results" / "round5_narrative.py")
    args = ap.parse_args()
    ns = {}
    exec(args.narrative.read_text(encoding="utf-8"), ns)
    src = args.out.with_suffix(".html")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(build_html(ns["NARRATIVE"]), encoding="utf-8")
    _render(src, args.out)
    print("→", args.out)


def _render(src, out):
    import subprocess
    import tempfile
    import time
    browser = r1.find_browser()
    profile = Path(tempfile.gettempdir()) / "forecast-pdf-profile"
    out.unlink(missing_ok=True)
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                    f"--print-to-pdf={out.resolve()}", src.resolve().as_uri()], check=True, capture_output=True, timeout=180)
    for _ in range(120):
        if out.exists() and out.stat().st_size > 0:
            size = out.stat().st_size
            time.sleep(1)
            if out.stat().st_size == size:
                break
        time.sleep(0.5)


if __name__ == "__main__":
    main()
