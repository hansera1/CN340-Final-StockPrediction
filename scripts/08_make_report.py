"""รายงานผลการรัน (HTML → PDF ด้วย Edge/Chrome headless) — ตัวเลขทุกตัวอ่านจากไฟล์ผลจริง

    python scripts/08_make_report.py [--title "รอบที่ 1"] [--out outputs/reports/report_run1.pdf]

ต้องรัน 01b_validate_sources.py และ 03_run_experiment.py (E0–E7s1) ก่อน
"""
import argparse
import html
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, RAW_DIR, load_experiments, load_rules
from forecast.evaluation.bootstrap import compare

EXP = OUTPUT_DIR / "experiments"
REP = OUTPUT_DIR / "reports"

# (การทดลอง, variant, ชื่อในรายงาน) — แถวหลักของตารางผล เรียงตามลำดับการทดลอง
MAIN = [("E0", "baseline", "E0 baseline (base rate + quantile เชิงประจักษ์)"),
        ("E1", "independent", "E1 Logistic + quantile GBM แยกหุ้น/รอบ"),
        ("E2", "cascade", "E2 cascade (ทิศทาง → ราคา)"),
        ("E6", "pooled", "E6 pooled แชร์ข้อมูลข้ามหุ้น"),
        ("E7", "e7a_scalogram_s0", "E7a ResNet18 + scalogram ขั้น 0"),
        ("E7", "e7b_candle_s0", "E7b ResNet18 + แท่งเทียน ขั้น 0"),
        ("E7s1", "e7a_scalogram_s1", "E7a ResNet18 + scalogram ขั้น 1 (5 seed)"),
        ("E7s1", "e7b_candle_s1", "E7b ResNet18 + แท่งเทียน ขั้น 1 (5 seed)")]
PILOT = [("E3", ["res_60m", "res_30m", "res_15m", "res_5m"]), ("E4", ["independent", "cnn1d"]),
         ("E5", ["independent", "spectral"])]
GROUPS = ["KBANK/morning", "KBANK/afternoon", "ADVANC/morning", "ADVANC/afternoon"]
GROUP_TH = {"KBANK/morning": "KBANK เช้า", "KBANK/afternoon": "KBANK บ่าย",
            "ADVANC/morning": "ADVANC เช้า", "ADVANC/afternoon": "ADVANC บ่าย", "ALL": "รวม"}


def esc(x):
    return html.escape(str(x))


def pct(x):
    return "–" if pd.isna(x) else f"{100 * x:.1f}%"


def num(x, d=3):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def preds(code, name, tz):
    p = pd.read_csv(EXP / code / f"{name}_predictions.csv")
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert(tz)
    return p


def summary_row(code, name, group="ALL"):
    s = pd.read_csv(EXP / code / "summary.csv")
    return s[(s["variant"] == name) & (s["group"] == group)].iloc[0]


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def forest_svg(rows, width=640):
    """ผลต่าง log loss เทียบ baseline + 95% CI ต่อโมเดล (ซ้าย = ดีกว่า baseline)"""
    lab_w, pad, row_h, top, right = 250, 16, 30, 26, 56
    plot_w = width - lab_w - pad - right
    lo = min(0, min(r[2] for r in rows)) - 0.005
    hi = max(r[3] for r in rows) + 0.005
    x = lambda v: lab_w + pad + (v - lo) / (hi - lo) * plot_w
    h = top + row_h * len(rows) + 34
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" '
           f'aria-label="ผลต่าง log loss เทียบ baseline พร้อมช่วงความเชื่อมั่น 95%">']
    for t in np.arange(np.ceil(lo * 100) / 100, hi + 1e-9, 0.01):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{h - 28}" class="grid"/>')
        out.append(f'<text x="{x(t):.1f}" y="{h - 12}" class="tick" text-anchor="middle">{0 if abs(t) < 1e-9 else t:+.2f}</text>')
    out.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 10}" y2="{h - 28}" class="zero"/>')
    out.append(f'<text x="{x(0) + 4:.1f}" y="{top - 12}" class="tick">baseline</text>')
    for i, (name, mean, l, u) in enumerate(rows):
        y = top + i * row_h + row_h / 2
        out.append(f'<text x="{lab_w}" y="{y + 4:.1f}" class="lab" text-anchor="end">{esc(name)}</text>')
        out.append(f'<line x1="{x(l):.1f}" x2="{x(u):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci"/>')
        out.append(f'<circle cx="{x(mean):.1f}" cy="{y:.1f}" r="5" class="dot"/>')
        out.append(f'<text x="{x(u) + 8:.1f}" y="{y + 4:.1f}" class="val">{mean:+.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def data_section(rules):
    tz = rules["timezone"]
    cv = pd.read_csv(REP / "validate_coarse_vs_1m.csv")
    names = {"open_am": "open แท่งแรกภาคเช้า = ราคาประมูลเปิด", "close_1229": "close แท่งล่าสุด ≤ 12:29",
             "open_pm": "open แท่งแรกภาคบ่าย = ราคาประมูลเปิดบ่าย", "close_1629": "close แท่งล่าสุด ≤ 16:29",
             "vol_last_excl_atc": "volume แท่งสุดท้ายไม่รวมประมูลปิด (ATC)"}
    rows = []
    for check, label in names.items():
        r = [label]
        for src in ("settrade 30m", "yfinance 60m"):
            g = cv[(cv["source"] == src) & (cv["check"] == check)]
            note = ""
            if check == "vol_last_excl_atc" and src == "yfinance 60m":
                note = f" (ต่าง {g['diff'].abs().min():,.0f}–{g['diff'].abs().max():,.0f} หุ้น; แบบรวม ATC ห่างกว่ามาก)"
            r.append(f"{int(g['match'].sum())}/{len(g)}{note}")
        rows.append(r)
    days = ", ".join(sorted(cv["date"].astype(str).unique()))
    check_tbl = table(["ตรวจ", "Settrade 30m", "yfinance 60m"], rows)

    # open แท่ง 10:00 ของ yfinance เทียบ open รายวันของ Settrade (= ราคาประมูลเปิด) และคุณภาพ yfinance 1d
    mism, bad1d = [], []
    for sym in rules["symbols"]:
        s = pd.read_csv(RAW_DIR / "settrade" / f"{sym}_1d.csv", parse_dates=["timestamp"]).set_index("timestamp")
        y = pd.read_csv(RAW_DIR / "yfinance" / f"{sym}_1d.csv", parse_dates=["timestamp"]).set_index("timestamp")
        h = pd.read_csv(RAW_DIR / "yfinance" / f"{sym}_60m.csv", parse_dates=["timestamp"])
        h = h[h["timestamp"].dt.strftime("%H:%M") == "10:00"]
        h10 = pd.Series(h["open"].to_numpy(), index=h["timestamp"].dt.normalize())
        j = pd.concat({"st": s["open"], "h10": h10}, axis=1).dropna()
        mism.append((sym, (~np.isclose(j["st"], j["h10"])).mean(), len(j)))
        k = s[["open"]].join(y[["open", "volume"]], rsuffix="_yf", how="inner")
        bad = k[~np.isclose(k["open"], k["open_yf"])]
        bad1d.append((sym, len(bad), int((bad["volume"] == 0).sum()), len(k)))

    ds = pd.read_pickle(sorted((Path(__file__).resolve().parents[1] / "data" / "processed")
                               .glob("dataset_yfinance_60m.pkl"))[0])
    lab = ds[ds["up"].notna()]
    flat = lab.groupby(["symbol", "session"]).agg(n=("direction", "size"),
                                                  flat=("direction", lambda d: (d == 0).mean()),
                                                  up=("direction", lambda d: (d == 1).mean()),
                                                  down=("direction", lambda d: (d == -1).mean()))
    flat_rows = [[GROUP_TH[f"{s}/{p}"], f"{int(r.n):,}", pct(r.up), pct(r.flat), pct(r.down)]
                 for (s, p), r in flat.reindex([tuple(g.split("/")) for g in GROUPS]).iterrows()]
    span = f"{lab['date'].min():%d/%m/%Y} – {lab['date'].max():%d/%m/%Y}"

    return f"""
<h2>2. ข้อมูล</h2>
<h3>2.1 แหล่งข้อมูลและความยาวย้อนหลัง</h3>
{table(["แหล่ง", "timeframe", "ย้อนหลังได้ (ตรวจ 30 ต.ค. 2026)", "ใช้ทำอะไร"], [
    ["Settrade Open API", "1m", "4 วันทำการ (ตั้งแต่ 26 ต.ค. 2026)", "วันจริง (แหล่งหลัก) + ตรวจความถูกต้อง"],
    ["Settrade Open API", "30m", "32 วันทำการ (ตั้งแต่ 11 ก.ย. 2026)", "ตรวจความถูกต้อง"],
    ["Settrade Open API", "1d", "~3 ปี (ตั้งแต่ 30 ต.ค. 2023)", "P(10:00) = open รายวัน (ราคาประมูลเปิด)"],
    ["yfinance", "60m", "729 วันทำการ (ตั้งแต่ 24 ต.ค. 2023)", "<b>ข้อมูลหลักของ backtest</b> (labels + features)"],
    ["yfinance", "5m / 15m / 30m", "60 วันทำการ", "E3–E5 (pilot)"],
    ["yfinance", "1m", "~8 วันทำการ", "วันจริง (สำรอง)"],
])}
<p>Settrade ระบุช่วงวันที่ (start/end) และขอย้อนหลังต่อเนื่องได้ แต่ระบบเก็บข้อมูลไว้เป็นช่วงคงที่ วันที่เก่ากว่านั้นได้ 0 แท่ง
จึงไม่พอทำ backtest ตามแผน (ต้องการฝึกอย่างน้อย 120 วัน) → ใช้ <b>yfinance 60m</b> เป็นข้อมูลหลัก และตรวจเทียบกับ Settrade 1m</p>

<h3>2.2 ตรวจแท่งหยาบเทียบ Settrade 1m (วันที่ซ้อนกัน: {esc(days)}, 2 หุ้น)</h3>
{check_tbl}
<ul>
<li>Settrade 30m ตรงทุกข้อ (มีแท่ง 09:30 / 13:30 สำหรับประมูลเปิด และแท่ง 16:30 = ATC แยก) แต่ย้อนได้แค่ 32 วัน</li>
<li>yfinance 60m: ราคา 12:30, 14:00, 16:30 ตรงทุกวัน; ราคา 16:30 = <b>ราคาซื้อขายล่าสุด ≤ 16:29 ไม่ใช่ ATC</b>
    (ใน 3 วันที่สองราคานี้ต่างกัน ตรงกับ 16:29 ทั้ง 3 วัน)</li>
<li><b>ปัญหาที่พบ:</b> open แท่ง 10:00 ของ yfinance คือราคา ณ 10:00 ไม่ใช่ราคาประมูลเปิด — ย้อน 2 ปีต่างกัน
    {", ".join(f"{s} {100 * m:.0f}% ของ {n} วัน" for s, m, n in mism)}
    → <b>แก้แล้ว:</b> P(10:00) ใช้ open รายวันของ Settrade (ตรงกับราคาประมูลเปิดทุกวันที่ตรวจ) หลังแก้ ราคาจุด label ตรง 32/32</li>
<li>ไม่ใช้ yfinance 1d: open ไม่ตรง Settrade {", ".join(f"{s} {b} จาก {n} วัน" for s, b, z, n in bad1d)}
    ส่วนใหญ่ปี 2024 (ราคาแบนและ volume 0) ทั้งที่แท่ง 60m ของวันเดียวกันปกติ</li>
<li>volume ช่วงเช้าของ yfinance เสีย (แท่ง 10:00 มักเป็น 0) → ตัด feature volume สะสมภายในวันออก
    หลังตัด features รอบบ่ายจาก backtest (60m) ตรงกับเส้นทางวันจริง (1m → รวมเป็น 60m) 100% ทั้ง Settrade และ yfinance</li>
</ul>

<h3>2.3 การกระจายของ label ({esc(span)})</h3>
{table(["หุ้น / รอบ", "จำนวนรอบ", "ขึ้น", "เท่าเดิม (flat)", "ลง"], flat_rows)}
<p>flat (ราคาปลายช่วง = ต้นช่วงพอดี) มีมาก เพราะช่วงราคา 100–400 บาท ช่องราคาคือ 0.50–1.00 บาท
→ <b>วิธีนับ flat เปลี่ยนตัวเลขความแม่นมาก ต้องยืนยันกับอาจารย์</b></p>
"""


def results_section(rules):
    tz = rules["timezone"]
    q = rules["quantiles"]
    base = preds("E0", "baseline", tz)
    e1 = preds("E1", "independent", tz)
    rows, forest = [], []
    for code, name, label in MAIN:
        s = summary_row(code, name)
        p = preds(code, name, tz)
        if (code, name) == ("E0", "baseline"):
            vs_base = "—"
        else:
            c = compare(p, base, q, block_days=rules["evaluation"]["block_days"], n_boot=2000)
            r = c[c["metric"] == "logloss"].iloc[0]
            vs_base = f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
            forest.append((label.split(" (")[0], r.mean_diff, r.ci_low, r.ci_high))
        rows.append([esc(label), num(s["logloss"]), pct(s["acc_flat_excluded"]), pct(s["acc_flat_as_wrong"]),
                     num(s["balanced_accuracy"]), num(s["mae_baht"], 2), pct(s["coverage80"]), vs_base])
    main_tbl = table(["โมเดล", "log loss ↓", "แม่น (ไม่นับ flat)", "แม่น (flat = ผิด)", "balanced acc",
                      "MAE (บาท) ↓", "coverage 80%", "log loss − baseline [95% CI]"], rows, "num")

    # เทียบ E1 (ตัวเทียบตามเงื่อนไขของ E2 / E6 / E7)
    vs1 = []
    for code, name, label in MAIN[2:]:
        c = compare(preds(code, name, tz), e1, q, n_boot=2000)
        r = c[c["metric"] == "logloss"].iloc[0]
        same = max(abs(r.ci_low), abs(r.ci_high)) < 5e-7
        vs1.append([esc(label.split(" (")[0]), f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                    "เท่ากัน (cascade ไม่เปลี่ยนขั้นทิศทาง)" if same else
                    "ดีกว่า" if r.ci_high < 0 else "แย่กว่า" if r.ci_low > 0 else "ไม่ต่างอย่างมีนัยสำคัญ"])

    seeds = []
    for name in ("e7a_scalogram_s1", "e7b_candle_s1"):
        p = preds("E7s1", name, tz)
        from forecast.evaluation.metrics import row_losses
        ll = [row_losses(p.assign(p_up=p[c]), q)["logloss"].mean() for c in p if c.startswith("p_up_seed")]
        seeds.append([name, f"{np.mean(ll):.3f}", f"{min(ll):.3f} – {max(ll):.3f}",
                      num(summary_row("E7s1", name)["logloss"])])

    grp_rows = []
    for g in GROUPS:
        r = [GROUP_TH[g]]
        for code, name in [("E0", "baseline"), ("E1", "independent"), ("E6", "pooled"), ("E7s1", "e7a_scalogram_s1")]:
            s = summary_row(code, name, g)
            r.append(f"{num(s['logloss'])} / {pct(s['acc_flat_excluded'])}")
        r.append(f"{summary_row('E0', 'baseline', g)['mae_baht']:.2f}")
        grp_rows.append(r)

    pilot_rows = []
    for code, names in PILOT:
        b = pd.read_csv(EXP / code / "bootstrap.csv")
        for i, name in enumerate(names):
            s = summary_row(code, name)
            ci = "—"
            if i:
                r = b[(b["comparison"].str.startswith(name + " ")) & (b["metric"] == "logloss")].iloc[0]
                ci = f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
            pilot_rows.append([code, name, f"{int(s['n'])}", num(s["logloss"]), pct(s["acc_flat_excluded"]), ci])

    return f"""
<h2>4. ผลการทดลอง</h2>
<h3>4.1 ผลรวม (walk-forward, {int(summary_row('E0', 'baseline')['n']):,} รอบ จาก 2 หุ้น × 2 รอบ)</h3>
{main_tbl}
<p class="note">log loss และ MAE ยิ่งต่ำยิ่งดี; coverage 80% ควรใกล้ 80%. ช่วง [ ] คือ 95% CI จาก block bootstrap รายวัน
(2,000 ครั้ง) ค่าติดลบ = ดีกว่า baseline. E7 ทำนายเฉพาะทิศทาง ช่วงราคาใช้ของ baseline</p>

<figure>
<figcaption>ผลต่าง log loss เทียบ baseline (จุด = ค่าเฉลี่ย, เส้น = 95% CI; ขวาของเส้น baseline = แย่กว่า)</figcaption>
{forest_svg(forest)}
</figure>

<h3>4.2 เทียบกับ E1 (ตัวเทียบตามคำถามของ E2, E6, E7)</h3>
{table(["โมเดล", "log loss − E1 [95% CI]", "สรุป"], vs1)}

<h3>4.3 E7 ขั้น 1: ผลราย seed (ไม่เลือก seed ที่ดีที่สุด)</h3>
{table(["variant", "log loss เฉลี่ยราย seed", "ช่วง (5 seed)", "log loss ของค่าเฉลี่ย 5 seed"], seeds)}
<p>การเฉลี่ยความน่าจะเป็นจาก 5 seed ลด log loss จาก ~0.726 เหลือ ~0.700 — สอดคล้องกับข้อสรุปว่าปัญหาหลักคือความแปรปรวน (overfit)</p>

<h3>4.4 แยกหุ้น / รอบ (log loss / ความแม่นแบบไม่นับ flat)</h3>
{table(["หุ้น / รอบ", "E0 baseline", "E1", "E6 pooled", "E7a ขั้น 1", "MAE baseline (บาท)"], grp_rows)}

<h3>4.5 E3–E5: pilot (yfinance 5m มีแค่ 60 วัน)</h3>
<p class="warn">⚠️ PILOT — ทดสอบได้แค่ 20 วัน (40–80 รอบ) ฝึกด้วยข้อมูล 30 วัน ใช้ดูทิศทางเท่านั้น ไม่ใช้ตัดสินเลือกวิธี</p>
{table(["การทดลอง", "variant", "รอบ", "log loss", "แม่น (ไม่นับ flat)", "เทียบ variant แรก [95% CI]"], pilot_rows)}
<p>ทุก variant มี log loss 0.63–1.11 และ CI กว้างมาก ยังสรุปเรื่องความละเอียด (E3), ลำดับข้อมูล (E4) และเวลา–ความถี่ (E5) ไม่ได้.
E4: cnn1d ชนะ independent อย่างมีนัยสำคัญ แต่เพราะ independent พังเมื่อฝึกด้วยข้อมูลน้อย (log loss 1.11) ไม่ใช่เพราะ cnn1d ดี
(0.629 ใกล้ base rate)</p>
"""


FEATURE_GROUPS = [
    ("รายวัน (ค่า ณ สิ้นวันทำการก่อนหน้า)", ["dow", "ret_1d", "ret_5d", "ret_20d", "rv_5d", "rv_20d", "rvol_1d", "dist_ma20"],
     "วันในสัปดาห์, ผลตอบแทน 1/5/20 วัน, ความผันผวน 5/20 วัน, volume เทียบค่าเฉลี่ย 20 วัน, ระยะห่างจาก MA20", "เช้า + บ่าย"),
    ("ภายในวัน (แท่งที่จบก่อนจุดตัด)", ["r_open_to_cut", "gap_open", "r_last30", "rv_intra", "absr_intra", "hl_range_intra"],
     "ผลตอบแทนตั้งแต่เปิดถึงจุดตัด, gap เปิด, ผลตอบแทน 30 นาทีสุดท้าย, realized vol, |r| เฉลี่ย, ช่วง high–low",
     "บ่ายเท่านั้น (09:15 ตลาดยังไม่เปิด)"),
    ("ประวัติช่วงเวลาเดียวกัน", ["r_same_mean5", "r_same_mean10", "r_same_mean20", "r_prev_morning", "r_prev_afternoon"],
     "r เฉลี่ยของรอบเดียวกัน 5/10/20 วันก่อน, r ของรอบเช้า/บ่ายเมื่อวาน", "เช้า + บ่าย"),
]


def wf_stats(ds, ev):
    """ขนาดชุดฝึก/ทดสอบของ walk-forward ต่อรอบ (เท่ากันทั้งสองหุ้น)"""
    from forecast.evaluation.splits import walk_forward_splits
    out = {}
    for sess, g in ds[ds["up"].notna()].groupby("session"):
        g = g[g["symbol"] == g["symbol"].iloc[0]]
        sp = list(walk_forward_splits(g["date"].to_numpy(), ev["test_block_days"], ev["min_train_days"],
                                      ev["holdout_days"]))
        ud = np.unique(g["date"])
        d = lambda x: pd.Timestamp(x).strftime("%d/%m/%Y")
        out[sess] = dict(days=len(ud), first=d(ud[0]), last=d(ud[-1]), folds=len(sp), train0=len(sp[0][0]),
                         train1=len(sp[-1][0]), test0=d(sp[0][1][0]), test1=d(sp[-1][1][-1]),
                         hold0=d(ud[-ev["holdout_days"]]) if ev["holdout_days"] else "–")
    return out


def method_section(rules):
    import inspect
    from forecast.models.cnn1d import CNN1DModel
    from forecast.models.image_heads import FrozenHeadModel, FrozenLRModel
    from forecast.models.independent import IndependentModel, QuantileBank
    from forecast.evaluation.runner import experiment_rules
    from forecast.features.dataset import feature_cols

    ev = rules["evaluation"]
    exps = load_experiments()
    proc = Path(__file__).resolve().parents[1] / "data" / "processed"
    ds60 = pd.read_pickle(proc / "dataset_yfinance_60m.pkl")
    ds5 = pd.read_pickle(proc / "dataset_yfinance_5m.pkl")
    ev5 = experiment_rules(rules, exps["E4"])["evaluation"]
    w60, w5 = wf_stats(ds60, ev), wf_stats(ds5, ev5)
    nf = {s: sum(g[c].notna().any() for c in feature_cols(g))
          for s, g in ds60[(ds60["symbol"] == "KBANK") & ds60["up"].notna()].groupby("session")}

    C = inspect.signature(IndependentModel).parameters["C"].default
    qb = QuantileBank([0.5]).params
    cnn = CNN1DModel(rules["quantiles"])
    head = FrozenHeadModel(rules["quantiles"])
    c0 = inspect.signature(FrozenLRModel).parameters["C"].default
    excl = ", ".join(rules.get("features", {}).get("exclude", [])) or "–"

    feat_rows = [[g, f"{len(n)}", esc(desc), when] for g, n, desc, when in FEATURE_GROUPS]
    feat_rows.append(["ของหุ้นอีกตัว (f_x_)", f"{sum(len(n) for _, n, _, _ in FEATURE_GROUPS) - 1}",
                      "ทุกกลุ่มข้างบนของอีกหุ้น ณ จุดตัดเดียวกัน (ยกเว้นวันในสัปดาห์)", "เช้า + บ่าย"])

    split_rows = []
    for name, w, note in [("A: yfinance 60m (E0–E2, E6, E7)", w60, ""),
                          ("B: yfinance 5m (E3–E5, pilot)", w5, "")]:
        for sess in ("morning", "afternoon"):
            s = w[sess]
            split_rows.append([name if sess == "morning" else "", "เช้า" if sess == "morning" else "บ่าย",
                               f"{s['days']} ({s['first']}–{s['last']})", f"{s['folds']}",
                               f"{s['train0']} → {s['train1']}", f"{s['test0']} – {s['test1']}", f"{s['hold0']} – {s['last']}"])

    hgb = (f"HistGradientBoosting quantile loss ×{len(rules['quantiles'])} (q = "
           f"{', '.join(f'{q:g}' for q in rules['quantiles'])}; depth {qb['max_depth']}, lr {qb['learning_rate']}, "
           f"{qb['max_iter']} รอบ, min leaf {qb['min_samples_leaf']}) — จัดการค่าหายเอง")
    logit = f"Logistic regression (เติมค่าหายด้วย median → standardize, C = {C})"
    train_rows = [
        ["E0 baseline", "A", "ไม่ใช้ features",
         "P(ขึ้น) = สัดส่วน 'ขึ้น' ในชุดฝึก; ช่วงราคา = quantile ของ z ในชุดฝึก เลื่อนให้ค่ากลาง = ราคาล่าสุดที่รู้ (z = 0)"],
        ["E1 independent", "A", f"features {nf['morning']} (เช้า) / {nf['afternoon']} (บ่าย)",
         f"ทิศทาง: {logit}; ราคา: {hgb}; แยกโมเดลต่อหุ้น × รอบ (4 ชุด)"],
        ["E2 cascade", "A", "เหมือน E1 + P(ขึ้น) จากขั้นแรก",
         "ขั้น 1 = Logistic แบบ E1; ขั้น 2 = quantile GBM ที่รับ P(ขึ้น) แบบ temporal out-of-fold (แบ่งชุดฝึก 5 บล็อกตามเวลา "
         "บล็อก i ได้ค่าจากโมเดลที่ฝึกบนบล็อกก่อนหน้า) ไม่ calibrate"],
        ["E3 ความละเอียด", "B", "features แบบ E1 แต่รวมแท่งเป็น 5 / 15 / 30 / 60 นาที; labels เดียวกันจาก 5m",
         "เหมือน E1; ใช้เฉพาะวันที่ข้อมูลทุกความละเอียดซ้อนกัน"],
        ["E4 1D-CNN (pilot)", "B (รอบบ่ายเท่านั้น)",
         "ลำดับ 120 แท่ง 5m ก่อนจุดตัด × 3 ช่อง (r, |r|, log volume มาตรฐาน)",
         f"conv1d 2 ชั้น (16 ช่อง, kernel 5) + dropout 0.3 → linear; Adam lr {cnn.lr}, weight decay {cnn.wd}, "
         f"batch {cnn.bs}, ≤ {cnn.epochs} epoch, early stopping patience {cnn.patience} บน {int(cnn.val_frac * 100)}% "
         f"วันท้ายของชุดฝึก; เฉลี่ย {len(cnn.seeds)} seed; ช่วงราคาใช้ baseline. เทียบกับ E1 บนรอบบ่ายชุดเดียวกัน"],
        ["E5 เวลา–ความถี่ (pilot)", "B", "features แบบ E1 + band power (Welch) ของผลตอบแทนก่อนจุดตัด", "เหมือน E1"],
        ["E6 pooled", "A", "features แบบ E1 + ตัวแปรบอกหุ้น", "เหมือน E1 แต่ฝึกโมเดลเดียวต่อรอบ รวมข้อมูลสองหุ้น (2 ชุดแทน 4)"],
        ["E7 ขั้น 0", "A", "ภาพ 224×224 จาก 56 แท่ง 60m ก่อนจุดตัด → ResNet18 (ImageNet, freeze, โหมด eval) → 512 feature",
         f"standardize → Logistic regression C = {c0}; ช่วงราคาใช้ baseline"],
        ["E7 ขั้น 1", "A", "embedding 512 มิติชุดเดียวกับขั้น 0",
         f"หัว dropout {head.dropout} + linear; AdamW lr {head.lr}, weight decay {head.wd}, batch {head.bs}; "
         f"warmup {head.warmup} epoch แล้ว cosine; ≤ {head.epochs} epoch, early stopping patience {head.patience} "
         f"บน {int(head.val_frac * 100)}% วันท้ายของชุดฝึก (log loss); {len(head.seeds)} seed เฉลี่ยความน่าจะเป็น"],
    ]
    return f"""
<h2>3. วิธีการ</h2>
<h3>3.1 Label และจุดตัดข้อมูล</h3>
<ul>
<li>r = log[P(ปลาย)/P(ต้น)] → ขึ้น / เท่าเดิม / ลง (flat = ต่าง 0 ช่องราคา). โมเดลทายทิศทางเป็น P(ขึ้น) โดย flat นับเป็น "ไม่ขึ้น"</li>
<li>P(10:00), P(14:00) = ราคาประมูลเปิดของภาค (เช้าใช้ open รายวันของ Settrade); P(12:30), P(16:30) = ราคาซื้อขายล่าสุด
    ≤ 12:29 / ≤ 16:29 ไม่รวม ATC. ราคา: z = log[P(ปลาย)/P(c)] ทำนาย quantile 10/25/50/75/90 แล้วแปลงกลับเป็นบาท</li>
<li>จุดตัด (ชั่วคราว) เช้า {rules['sessions']['morning']['cutoff']}, บ่าย {rules['sessions']['afternoon']['cutoff']}:
    ใช้เฉพาะแท่งที่จบก่อนจุดตัด (แท่ง 12:00–12:30 ไม่ถูกใช้รอบบ่าย) ทั้ง backtest และวันจริง — มีเทสต์ยืนยัน</li>
</ul>

<h3>3.2 Features (ชุดข้อมูล A: เช้า {nf['morning']} ตัว, บ่าย {nf['afternoon']} ตัว)</h3>
{table(["กลุ่ม", "จำนวน", "ประกอบด้วย", "ใช้ในรอบ"], feat_rows)}
<p class="note">ตัดออก: {esc(excl)} (volume ช่วงเช้าของ yfinance เสีย). ยังไม่มีข้อมูลภายนอก (ตลาดต่างประเทศ ค่าเงิน ฟลอว์)</p>

<h3>3.3 ชุดข้อมูลและการแบ่ง train / test (walk-forward แบบขยายหน้าต่าง)</h3>
{table(["ชุดข้อมูล", "รอบ", "วันที่มี label", "จำนวน fold", "วันฝึก (fold แรก → สุดท้าย)", "ช่วงทดสอบ", "holdout (ไม่แตะ)"],
       split_rows)}
<p class="note">ทุก fold ฝึกด้วยทุกวันก่อนหน้า แล้วทดสอบ {ev['test_block_days']} วันถัดไป เลื่อนไปเรื่อย ๆ (ไม่สุ่มแถวข้ามเวลา);
ชุด A ฝึกขั้นต่ำ {ev['min_train_days']} วัน, ชุด B {ev5['min_train_days']} วัน. รอบบ่ายเริ่มที่ 25/03/2024 (ภาคบ่ายเปิด 14:00).
ช่วง holdout {ev['holdout_days']} วันสุดท้ายเก็บไว้ประเมินครั้งเดียวหลังล็อกวิธี. เทียบโมเดลด้วย block bootstrap รายวัน 2,000 ครั้ง</p>

<h3>3.4 ข้อมูลและการฝึกของแต่ละการทดลอง</h3>
{table(["การทดลอง", "ชุดข้อมูล", "input ของโมเดล", "โมเดลและการฝึก"], train_rows, "train")}
<p class="note">ทุกการทดลองฝึกโมเดลใหม่ทุก fold จากข้อมูลก่อนช่วงทดสอบเท่านั้น; hyperparameter ทุกตัวตั้งไว้ก่อนและไม่ได้จูนจากผลทดสอบ</p>
"""


def build_html(rules, title):
    ev = rules["evaluation"]
    exps = load_experiments()
    best = summary_row("E0", "baseline")
    e1 = summary_row("E1", "independent")
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<title>{esc(title)}</title>
<style>
:root {{ --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#8a8984;
        --rule:#e2e1dc; --series-1:#2a78d6; --accent-bg:#f3f6fb; --warn-bg:#fff6e5; }}
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
body {{ font-family: "Sarabun", "Leelawadee UI", "Tahoma", sans-serif; color: var(--text-primary);
        background: #fff; font-size: 10.5pt; line-height: 1.55; margin: 0; }}
h1 {{ font-size: 20pt; margin: 0 0 2mm; }}
h2 {{ font-size: 14pt; margin: 7mm 0 2mm; border-bottom: 1.5px solid var(--text-primary); padding-bottom: 1mm;
      break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin: 4mm 0 1.5mm; break-after: avoid; }}
.sub {{ color: var(--text-secondary); margin-bottom: 4mm; }}
table {{ border-collapse: collapse; width: 100%; margin: 1.5mm 0 2.5mm; font-size: 9.3pt; break-inside: avoid; }}
th {{ text-align: left; font-weight: 600; color: var(--text-secondary); border-bottom: 1px solid var(--text-primary);
      padding: 1.2mm 1.5mm; vertical-align: bottom; }}
td {{ border-bottom: 1px solid var(--rule); padding: 1.1mm 1.5mm; vertical-align: top; }}
table.num td:not(:first-child) {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
table.train td:first-child {{ white-space: nowrap; font-weight: 600; }} table.train td:nth-child(2) {{ white-space: nowrap; }}
.box {{ background: var(--accent-bg); border-left: 3px solid var(--series-1); padding: 2.5mm 4mm; margin: 2mm 0 3mm;
        break-inside: avoid; }}
.warn {{ background: var(--warn-bg); padding: 1.5mm 3mm; }}
.note {{ color: var(--text-secondary); font-size: 9pt; }}
ul {{ margin: 1mm 0 2mm; padding-left: 6mm; }} li {{ margin: 0.6mm 0; }}
figure {{ margin: 2mm 0 3mm; break-inside: avoid; }}
figcaption {{ font-size: 9.5pt; color: var(--text-secondary); margin-bottom: 1mm; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg .zero {{ stroke: var(--text-secondary); stroke-width: 1.5; stroke-dasharray: 4 3; }}
svg .ci {{ stroke: var(--series-1); stroke-width: 2; stroke-linecap: round; }}
svg .dot {{ fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; }}
svg text {{ font-family: inherit; }} svg .lab {{ font-size: 11px; fill: var(--text-primary); }}
svg .tick, svg .val {{ font-size: 10px; fill: var(--text-secondary); }}
.pb {{ break-before: page; }}
</style></head><body>
<h1>{esc(title)}</h1>
<div class="sub">พยากรณ์ทิศทางและราคา KBANK / ADVANC รอบเช้า (10:00→12:30) และรอบบ่าย (14:00→16:30)
· สร้างเมื่อ {datetime.now():%d/%m/%Y %H:%M}</div>

<h2>1. สรุป</h2>
<div class="box">
<ul>
<li><b>ยังไม่มีโมเดลใดชนะ baseline</b> — baseline ได้ log loss {best['logloss']:.3f};
    โมเดลที่ดีที่สุดรองลงมา (E7 ขั้น 1, ~0.700) และ E6 pooled (0.705) แย่กว่า baseline อย่างมีนัยสำคัญ</li>
<li>โมเดลทายทิศทางถูกขึ้นเล็กน้อย ({pct(e1['acc_flat_excluded'])} เทียบ {pct(best['acc_flat_excluded'])} แบบไม่นับ flat)
    แต่ให้ความน่าจะเป็นมั่นใจเกินจริงและช่วงราคาแคบเกิน (coverage 80% ของ E1 = {pct(e1['coverage80'])})
    → อาการ overfit: 25 (เช้า) / 37 (บ่าย) features ต่อข้อมูลฝึก 120–715 วันต่อหุ้น/รอบ และสัญญาณอ่อนมาก</li>
<li>ทิศทางที่ช่วยได้ชัด: ลดความแปรปรวน — แชร์ข้อมูลข้ามหุ้น (E6) และเฉลี่ยหลาย seed (E7) ดีกว่า E1 แต่ยังไม่พอชนะ baseline</li>
<li>cascade (E2) ไม่ช่วย; E3–E5 (pilot 60 วัน) ข้อมูลน้อยเกินสรุป</li>
<li><b>ข้อเสนอ:</b> ล็อก E0 baseline เป็นตัวส่งจริง ถ้ายังมีเวลาก่อน 5 พ.ย. ทำการทดลองใหม่ 1 ชุดที่กำหนดล่วงหน้า
    (pooled + regularization แรงขึ้น + features น้อยลง + calibration) แล้วตัดสินด้วย holdout ครั้งเดียว</li>
</ul></div>

{data_section(rules)}

{method_section(rules)}

{results_section(rules)}

<h2>5. ข้อจำกัด</h2>
<ul>
<li>ข้อมูลหลักมาจาก yfinance (ไม่ใช่แหล่งทางการ) ตรวจเทียบ Settrade 1m ได้แค่ 4 วัน; volume ช่วงเช้าเชื่อไม่ได้</li>
<li>ยังไม่มีข้อมูลภายนอก (ตลาดต่างประเทศ ค่าเงิน ฟลอว์) — features ใช้แค่ราคา/volume ของสองหุ้น</li>
<li>จุดตัด 09:15 / 12:15 เป็นค่าชั่วคราว ยังไม่ได้วัดความหน่วงจริงช่วงตลาดเปิด</li>
<li>ผลทั้งหมดเป็น backtest ที่ยังไม่ได้แตะช่วง holdout; E3–E5 เป็น pilot</li>
</ul>

<h2>6. ขั้นต่อไปและคำถามถึงอาจารย์</h2>
<ul>
<li>ตัดสินโมเดลที่ล็อก (ข้อเสนอ: E0 baseline) ภายใน 5 พ.ย.; รันช่วง holdout ครั้งเดียวหลังล็อก</li>
<li>วัดความหน่วงช่วงตลาดเปิดบนเครื่องที่ deploy แล้วตั้งจุดตัดจริง; เริ่มสะสมข้อมูล 1m ทุกวันหลัง 17:00</li>
<li>E7 ขั้น 2 (ปลด layer4) ผ่านเงื่อนไขตามแผน แต่ยังแพ้ baseline — ไม่แนะนำใช้เวลา; E8 (Chronos-2) ยังไม่เริ่ม</li>
<li><b>ถามอาจารย์:</b> (1) ราคา 16:30 ใช้ราคาซื้อขายล่าสุดหรือ ATC (2) flat นับถูก ผิด หรือไม่นับ
    (3) รูปแบบการส่งผลและการปัดราคาตาม tick (4) ขอบเขตโมเดลที่อนุญาต (รวม pretrained)</li>
</ul>
</body></html>"""


def find_browser():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(p).exists():
            return p
    return shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="รายงานผลการรันรอบที่ 1")
    ap.add_argument("--out", type=Path, default=REP / "report_run1.pdf")
    args = ap.parse_args()
    rules = load_rules()
    src = args.out.with_suffix(".html")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(build_html(rules, args.title), encoding="utf-8")
    browser = find_browser()
    if not browser:
        raise SystemExit(f"ไม่พบ Edge/Chrome — เปิด {src} แล้วสั่งพิมพ์เป็น PDF เอง")
    # profile แยก — ไม่งั้น Edge/Chrome ที่เปิดอยู่จะรับงานไปเองและไม่สร้างไฟล์
    profile = Path(tempfile.gettempdir()) / "forecast-pdf-profile"
    args.out.unlink(missing_ok=True)
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={profile}", f"--print-to-pdf={args.out.resolve()}", src.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    for _ in range(120):          # ตัวเปิด Edge อาจจบก่อน process ลูกเขียนไฟล์เสร็จ → รอจนขนาดไฟล์นิ่ง
        if args.out.exists() and args.out.stat().st_size > 0:
            size = args.out.stat().st_size
            time.sleep(1)
            if args.out.stat().st_size == size:
                break
        time.sleep(0.5)
    if not args.out.exists():
        raise SystemExit(f"สร้าง PDF ไม่สำเร็จ — เปิด {src} แล้วสั่งพิมพ์เป็น PDF เอง")
    print("→", args.out)


if __name__ == "__main__":
    main()
