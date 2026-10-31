"""รายงานรอบที่ 3 (HTML → PDF) — ตัวเลขในตารางอ่านจาก results/ ทั้งหมด

    python experiments/16_make_report_round3.py [--out outputs/reports/report_round3.pdf]

ต้องรันก่อน: experiments/12_score_e12.py, 13_wavelet_coherence.py, 14_score_holdout.py, 15_e11_pilot.py (ถ้ามี)
"""
import argparse
import base64
import importlib
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from forecast.config import OUTPUT_DIR, ROOT

r2 = importlib.import_module("10_make_report_round2")
r1 = r2.r1
esc, table, pct, ci_pp, ci_ll, forest_svg = r1.esc, r1.table, r1.pct, r2.ci_pp, r2.ci_ll, r2.forest_svg
RES = ROOT / "results"
GROUP_TH = r1.GROUP_TH
SETS = {"f0": "F0", "fd": "F0 + F-daily", "fdh": "F0 + F-daily + F-60m"}
ALGO = {"lr": "LR", "rf": "RF", "xgb": "XGB", "mlp": "MLP"}


def img(path, width="100%"):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def label(m):
    if m in ("E0-ud", "E9-LRx"):
        return m
    fs, a = m.rsplit("_", 1)
    return f"E12 {SETS[fs]} · {ALGO[a]}"


def point_price_section():
    t = pd.read_csv(RES / "point_price_old_vs_new.csv")
    rows = [[GROUP_TH.get(f"{r.symbol}/{r.session}", f"{r.symbol} {r.session}"), esc(r.period), r.n,
             f"{r.mae_old:.3f}", f"{r.mae_new:.3f}", pct(r.same_price)] for r in t.itertuples()]
    return table(["หุ้น / รอบ", "ช่วงที่เทียบ", "รอบ", "MAE แบบเก่า (บาท)", "MAE แบบใหม่ (บาท)", "ราคาเท่ากัน"], rows, "small")


def e12_section():
    t = pd.read_csv(RES / "e12" / "e12_table.csv")
    a = t[t.group == "ALL"]
    rows = []
    for r in a.itertuples():
        d = r._asdict()
        rows.append([esc(label(r.model)), pct(r.acc_nf), f"{r.logloss_ud:.4f}",
                     ci_pp(d, "vsE0ud_d_acc_nf") if r.model != "E0-ud" else "–",
                     ci_pp(d, "vsLRx_d_acc_nf") if r.model not in ("E0-ud", "E9-LRx") else "–",
                     ci_ll(d, "vsLRx_d_logloss_ud") if r.model not in ("E0-ud", "E9-LRx") else "–"])
    main = table(["โมเดล", "แม่น (ไม่นับ flat)", "log loss ขึ้น/ลง", "Δ แม่น เทียบ E0-ud", "Δ แม่น เทียบ E9-LRx",
                  "Δ log loss เทียบ E9-LRx"], rows, "small")
    forest = forest_svg([(label(r.model), r.vsLRx_d_acc_nf, r.vsLRx_d_acc_nf_lo, r.vsLRx_d_acc_nf_hi)
                         for r in a.itertuples() if r.model not in ("E0-ud", "E9-LRx")])         .replace(">E0-ud (", ">E9-LRx (").replace("เทียบ E0-ud", "เทียบ E9-LRx")
    per = []
    for m in ["E9-LRx", "fd_lr", "fdh_lr"]:
        for g in ["KBANK/morning", "KBANK/afternoon", "ADVANC/morning", "ADVANC/afternoon"]:
            r = t[(t.model == m) & (t.group == g)].iloc[0].to_dict()
            per.append([esc(label(m)) if g == "KBANK/morning" else "", GROUP_TH[g], int(r["n_nonflat"]),
                        pct(r["acc_nf"]), ci_pp(r, "vsE0ud_d_acc_nf"),
                        ci_pp(r, "vsLRx_d_acc_nf") if m != "E9-LRx" else "–"])
    per_t = table(["โมเดล", "หุ้น / รอบ", "รอบไม่ flat", "แม่น", "Δ เทียบ E0-ud", "Δ เทียบ E9-LRx"], per, "small")
    imp = pd.read_csv(RES / "e12" / "importance.csv")
    blocks = []
    for (m, s), g in imp.groupby(["model", "session"], sort=False):
        g = g.sort_values("d_logloss_ud", ascending=False).head(6)
        rows = [[esc(f), f"{x:+.4f}", f"{100 * y:+.1f}"] for f, x, y in zip(g.feature, g.d_logloss_ud, g.d_acc_nf)]
        blocks.append(f"<div><b>{esc(label(m))} · {'เช้า' if s == 'morning' else 'บ่าย'}</b>" +
                      table(["feature", "Δ log loss", "Δ แม่น (จุด)"], rows, "small") + "</div>")
    conv = pd.read_csv(RES / "e12" / "mlp_convergence.csv")
    pilot = pd.read_csv(RES / "e12" / "pilot_timeframe.csv")
    prow = [[esc(r.timeframe), r.n_days, r.n_nonflat, pct(r.acc_nf), f"{r.logloss_ud:.4f}",
             ci_pp(r._asdict(), "d_acc_nf") if r.timeframe != "E0-ud" else "–"] for r in pilot.itertuples()]
    return (main, forest, per_t, '<div class="grid3">' + "".join(blocks) + "</div>",
            ", ".join(f"{label(r.model)} {pct(r.share_converged)}" for r in conv.itertuples()),
            table(["แท่งของ F-60m", "วันทดสอบ", "รอบไม่ flat", "แม่น", "log loss", "Δ แม่น เทียบ E0-ud"], prow, "small"))


def holdout_section():
    d = pd.read_csv(RES / "holdout" / "holdout_direction.csv")
    rows = []
    for r in d.itertuples():
        x = r._asdict()
        rows.append([esc(r.model) if r.group == "ALL" else "", "รวม" if r.group == "ALL" else GROUP_TH[r.group],
                     int(r.n_nonflat), pct(r.acc_nf), pct(r.acc_fw), f"{r.logloss_ud:.4f}",
                     ci_pp(x, "vsE0ud_d_acc_nf") if r.model != "E0-ud" else "–"])
    p = pd.read_csv(RES / "holdout" / "holdout_price.csv")
    prow = [["รวม" if r.group == "ALL" else GROUP_TH[r.group], r.n, pct(r.share_shifted), f"{r.mae_rule_baht:.3f}",
             f"{r.mae_pc_baht:.3f}", pct(r.coverage50), pct(r.coverage80)] for r in p.itertuples()]
    return (table(["โมเดล", "ชุด", "รอบไม่ flat", "แม่น (ไม่นับ flat)", "แม่น (flat = ผิด)", "log loss", "Δ แม่น เทียบ E0-ud"],
                  rows, "small"),
            table(["ชุด", "รอบ", "เลื่อนราคา", "MAE กฎ (บาท)", "MAE P(c) (บาท)", "ครอบ 50%", "ครอบ 80%"], prow, "small"))


def a1_section():
    s = pd.read_csv(RES / "a1" / "coherence_by_period.csv")
    g = s.groupby(["interval", "pair", "session"], sort=False).agg(
        n=("n_segments", "first"), coh=("coherence", "mean"), thr=("threshold95", "mean"),
        frac=("frac_sig_area", "mean")).reset_index()
    lag = pd.read_csv(RES / "a1" / "lagged_corr.csv").pivot_table(index=["interval", "pair", "session"],
                                                                   columns="lag", values="corr")
    rows = []
    for r in g.itertuples():
        l = lag.loc[(r.interval, r.pair, r.session)]
        rows.append([r.interval, esc(r.pair), "เช้า" if r.session == "morning" else "บ่าย", r.n, f"{r.coh:.3f}",
                     f"{r.thr:.3f}", pct(r.frac), f"{l[0]:+.3f}", f"{l[-1]:+.3f} / {l[1]:+.3f}"])
    return table(["แท่ง", "คู่", "รอบ", "ช่วง", "coherence เฉลี่ย", "เกณฑ์ 95% (AR1)", "พื้นที่มีนัยสำคัญ",
                  "corr lag 0", "corr lag −1 / +1"], rows, "small")


def e11_section():
    path = RES / "e11" / "e11_pilot.csv"
    if not path.exists():
        return "<p>ยังไม่มีผล</p>"
    t = pd.read_csv(path)
    rows = [[esc(r.model), r.n_days, r.n_nonflat, pct(r.acc_nf), f"{r.logloss_ud:.4f}",
             ci_pp(r._asdict(), "d_acc_nf") if r.model != "E0-ud" else "–"] for r in t.itertuples()]
    return table(["โมเดล", "วันทดสอบ", "รอบไม่ flat", "แม่น", "log loss", "Δ แม่น เทียบ E0-ud"], rows, "small")


def build_html(nar):
    e12_main, e12_forest, e12_per, e12_imp, conv, pilot = e12_section()
    h_dir, h_price = holdout_section()
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><title>รายงานรอบที่ 3</title>
<style>{r2.CSS}</style></head><body>
<h1>ผลการทดสอบรอบที่ 3: E12, ล็อกโมเดล, holdout และย่านความถี่สูง</h1>
<div class="sub">KBANK / ADVANC · walk-forward 24/04/2024 – 12/10/2026 · holdout 14–29/10/2026 (รันครั้งเดียว)
· สร้างเมื่อ {datetime.now():%d/%m/%Y %H:%M}</div>
{nar["summary"]}

<h2>1. ราคาจุดในวันจริง (ข้อ 1)</h2>
{nar["point"]}
{point_price_section()}
{nar["point_after"]}

<h2 class="pb">2. E12: สูตร feature และโมเดลจากงานรายวันเดิม (CN340)</h2>
{nar["e12_method"]}
<h3>2.1 ผลรวมทุกหุ้น/รอบ (แถวทดสอบเดียวกับ E9; ✱ = CI ไม่ครอบ 0)</h3>
{e12_main}
<figure><figcaption>ผลต่างความแม่น (ไม่นับ flat) ของ E12 เทียบ E9-LRx และ 95% CI (จุด %; ขวาของเส้นประ = ดีกว่า E9-LRx)</figcaption>
{e12_forest}</figure>
{nar["e12_result"]}
<h3>2.2 แยกหุ้น / รอบ ของตัวที่ผ่านเกณฑ์</h3>
{e12_per}
<h3>2.3 Permutation importance นอกตัวอย่าง (ตัวที่แม่นที่สุดในแต่ละชุด feature; 6 อันดับแรกตาม Δ log loss)</h3>
<p class="note">Δ = ค่าเมื่อสุ่มค่า feature นั้นใหม่จากชุดฝึกของแต่ละ fold ลบค่าจริง (บวก = feature มีประโยชน์) ·
MLP converge: {conv}</p>
{e12_imp}
{nar["e12_imp"]}
<h3>2.4 Pilot timeframe (ไม่ใช้เลือกโมเดล)</h3>
{pilot}
{nar["pilot"]}
<h3>2.5 Test set ของงาน CN340 เดิม (ข้อ 2.0)</h3>
{nar["cn340"]}

<h2 class="pb">3. โมเดลที่ล็อก (ข้อ 3 — results/lock_decision.md)</h2>
{nar["lock"]}

<h2>4. Holdout 10 วันสุดท้าย (ข้อ 4 — รันครั้งเดียว ไม่ปรับอะไรหลังเห็นผล)</h2>
<h3>4.1 ทิศทาง</h3>
{h_dir}
<h3>4.2 ราคา (กฎค่ามัธยฐาน)</h3>
{h_price}
{nar["holdout"]}

<h2 class="pb">5. ความสัมพันธ์ในย่านความถี่สูง (ข้อ 5 — สำหรับรายงาน ไม่ใช้เลือกโมเดล)</h2>
{nar["a1_method"]}
<h3>5.1 A1: สรุปตามคู่หุ้น</h3>
{a1_section()}
<figure>{img(RES / "a1" / "coherence_by_period.png")}</figure>
{nar["a1_result"]}
<h3>5.2 ตัวอย่างแผนที่ coherence รายวัน (KBANK–ADVANC, 1m, เช้า)</h3>
<figure>{img(RES / "a1" / "wtc_1m_KBANK_ADVANC_morning.png")}</figure>
<p class="note">แผนที่ทุกคู่ / ทุกรอบอยู่ใน results/a1/</p>
<h3 class="pb">5.3 A2: ความละเอียดของข้อมูลกับ Nyquist</h3>
<figure>{img(RES / "a2" / "resolution_compare.png")}</figure>
{nar["a2"]}

<h2>6. E11 PILOT: ResNet18 บน scalogram 5m (ข้อ 9)</h2>
{nar["e11_method"]}
{e11_section()}
{nar["e11_result"]}

<h2 class="pb">7. ระบบวันจริง (ข้อ 7)</h2>
{nar["live"]}

<h2>8. งานที่เหลือและความคาดหวัง</h2>
{nar["remaining"]}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "reports" / "report_round3.pdf")
    ap.add_argument("--narrative", type=Path, default=RES / "round3_narrative.py")
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
