"""รายงานรอบที่ 2 (HTML → PDF) — ตัวเลขทุกตัวอ่านจาก outputs/reports/round2/*.csv และ outputs/experiments/

    python scripts/09_score_round2.py          # คำนวณตัวชี้วัดก่อน
    python scripts/10_make_report_round2.py [--out outputs/reports/report_round2.pdf]

ส่วน E10 ที่ยังรันไม่เสร็จจะแสดงเท่าที่มี (รันซ้ำได้เมื่อมีผลเพิ่ม)
"""
import argparse
import importlib
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules

r1 = importlib.import_module("08_make_report")
esc, pct, num, table = r1.esc, r1.pct, r1.num, r1.table

EXP = OUTPUT_DIR / "experiments"
R2 = OUTPUT_DIR / "reports" / "round2"
LOGS = OUTPUT_DIR / "logs"
GROUPS = ["KBANK/morning", "KBANK/afternoon", "ADVANC/morning", "ADVANC/afternoon"]
GROUP_TH = r1.GROUP_TH
ARCH_TH = {"resnet18": "ResNet18", "densenet121": "DenseNet121", "inception_v3": "Inception-v3",
           "vgg16_bn": "VGG16-BN", "alexnet": "AlexNet"}


def pp(x):
    return "–" if pd.isna(x) else f"{100 * x:+.1f}"


def ci_pp(row, key):
    lo, hi = row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if pd.isna(lo):
        return "–"
    sig = " ✱" if lo > 0 or hi < 0 else ""
    return f"{pp(row[key])} [{pp(lo)}, {pp(hi)}]{sig}"


def ci_ll(row, key):
    lo, hi = row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if pd.isna(lo):
        return "–"
    sig = " ✱" if lo > 0 or hi < 0 else ""
    return f"{row[key]:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}"


def forest_svg(rows, width=660, unit="จุด %"):
    """ผลต่างความแม่น (ไม่นับ flat) เทียบ E0-ud + 95% CI (ขวา = ดีกว่า)"""
    lab_w, pad, row_h, top, right = 230, 14, 24, 26, 60
    plot_w = width - lab_w - pad - right
    lo = min(-0.02, min(r[2] for r in rows)) - 0.005
    hi = max(0.02, max(r[3] for r in rows)) + 0.005
    x = lambda v: lab_w + pad + (v - lo) / (hi - lo) * plot_w
    h = top + row_h * len(rows) + 34
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" aria-label="ผลต่างความแม่นเทียบ E0-ud">']
    for t in np.arange(np.ceil(lo * 50) / 50, hi + 1e-9, 0.02):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{h - 28}" class="grid"/>')
        out.append(f'<text x="{x(t):.1f}" y="{h - 12}" class="tick" text-anchor="middle">{100 * t:+.0f}</text>')
    out.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 10}" y2="{h - 28}" class="zero"/>')
    out.append(f'<text x="{x(0) + 4:.1f}" y="{top - 12}" class="tick">E0-ud ({unit})</text>')
    for i, (name, mean, l, u) in enumerate(rows):
        y = top + i * row_h + row_h / 2
        out.append(f'<text x="{lab_w}" y="{y + 4:.1f}" class="lab" text-anchor="end">{esc(name)}</text>')
        out.append(f'<line x1="{x(l):.1f}" x2="{x(u):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci"/>')
        out.append(f'<circle cx="{x(mean):.1f}" cy="{y:.1f}" r="4.5" class="dot"/>')
        out.append(f'<text x="{x(u) + 6:.1f}" y="{y + 4:.1f}" class="val">{100 * mean:+.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def curve_svg(c, title, width=320, height=170):
    """learning curve: ค่าเฉลี่ย train / validation loss ต่อ epoch (เฉลี่ยทุก fold และ seed ที่ยังฝึกอยู่)"""
    m = c.groupby("epoch")[["train_loss", "val_loss"]].mean()
    n = c.groupby("epoch").size()
    m = m[n >= n.iloc[0] * 0.5]          # แสดงเฉพาะ epoch ที่ยังมี fold × seed ฝึกอยู่อย่างน้อยครึ่งหนึ่ง
    l, r, t, b = 36, 10, 22, 26
    lo, hi = m.min().min() - 0.005, m.max().max() + 0.005
    X = lambda e: l + (e - 1) / max(1, m.index.max() - 1) * (width - l - r)
    Y = lambda v: t + (hi - v) / (hi - lo) * (height - t - b)
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{esc(title)}">',
           f'<text x="{l}" y="13" class="lab">{esc(title)}</text>']
    for v in np.linspace(lo, hi, 4):
        out.append(f'<line x1="{l}" x2="{width - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{l - 4}" y="{Y(v) + 3:.1f}" class="tick" text-anchor="end">{v:.3f}</text>')
    for e in m.index[::max(1, len(m) // 6)]:
        out.append(f'<text x="{X(e):.1f}" y="{height - 10}" class="tick" text-anchor="middle">{e}</text>')
    for col, cls in [("train_loss", "s2"), ("val_loss", "s1")]:
        pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in m[col].items())
        out.append(f'<polyline points="{pts}" class="line {cls}"/>')
    out.append("</svg>")
    return "".join(out)


def variant_times():
    """เวลาที่ใช้ต่อ variant จาก log ของ 03_run_experiment.py ('<variant>: เสร็จ HH:MM:SS (N วินาที)')"""
    out = {}
    for log in [LOGS / "run_E10.log"]:
        if log.exists():
            for m in re.finditer(r"^\s+(\S+): เสร็จ \S+ \((\d+) วินาที\)", log.read_text(encoding="utf-8"), re.M):
                out[m.group(1)] = max(out.get(m.group(1), 0), int(m.group(2)))   # --resume พิมพ์ซ้ำเป็น ~1 วินาที
    return out


def metrics():
    t = pd.read_csv(R2 / "metrics_round2.csv")
    return t


def round1_section(t):
    rows = []
    for _, r in t[t["group"] == "ALL"].iterrows():
        if r["exp"] in ("E9", "E10"):
            continue
        rows.append([esc(r["model"]), pct(r["acc_nf"]), pct(r["acc_fw"]), num(r["logloss_ud"], 4),
                     ci_pp(r, "vsE0_d_acc_nf"), ci_pp(r, "vsE0ud_d_acc_nf"), ci_ll(r, "vsE0ud_d_logloss_ud")])
    e0 = t[(t["model"] == "E0-ud baseline (ขึ้น vs ลง)") & (t["group"] == "ALL")].iloc[0]
    rows.append([esc(e0["model"]) + " <i>(รอบ 2)</i>", pct(e0["acc_nf"]), pct(e0["acc_fw"]),
                 num(e0["logloss_ud"], 4), ci_pp(e0, "vsE0_d_acc_nf"), "—", "—"])
    return table(["โมเดล", "แม่น (ไม่นับ flat)", "แม่น (flat = ผิด)", "log loss ขึ้น/ลง",
                  "Δแม่น vs E0 รอบแรก [CI]", "Δแม่น vs E0-ud [CI]", "Δlog loss vs E0-ud [CI]"], rows, "num")


def quarter_section():
    q = pd.read_csv(R2 / "direction_by_quarter.csv")
    q = q[q["n"] >= 30]
    quarters = sorted(q["quarter"].unique())
    rows = []
    for g in GROUPS:
        sym, sess = g.split("/")
        s = q[(q["symbol"] == sym) & (q["session"] == sess)].set_index("quarter")
        rows.append([GROUP_TH[g]] + [("–" if qq not in s.index else
                                      f'{100 * s.at[qq, "up_share_nonflat"]:.0f}') for qq in quarters])
    return table(["หุ้น / รอบ"] + [qq.replace("Q", " Q") for qq in quarters], rows, "num small")


def e9_section(t):
    rows = []
    for _, r in t[(t["exp"] == "E9") & (t["group"] == "ALL")].iterrows():
        rows.append([esc(r["model"]), pct(r["acc_nf"]), pct(r["acc_fw"]), num(r["logloss_ud"], 4),
                     ci_pp(r, "vsE0ud_d_acc_nf"), ci_ll(r, "vsE0ud_d_logloss_ud"), pct(r["pred_up_share"])])
    main = table(["โมเดล", "แม่น (ไม่นับ flat)", "แม่น (flat = ผิด)", "log loss ขึ้น/ลง",
                  "Δแม่น vs E0-ud [CI]", "Δlog loss vs E0-ud [CI]", "สัดส่วนที่ทายขึ้น"], rows, "num")
    rows = []
    for g in GROUPS:
        for _, r in t[(t["exp"] == "E9") & (t["group"] == g)].iterrows():
            rows.append([GROUP_TH[g] if r["variant"] == "baseline_ud" else "", esc(r["model"]),
                         f'{pct(r["acc_nf"])} / {num(r["logloss_ud"], 4)}', ci_pp(r, "vsE0ud_d_acc_nf"),
                         ci_ll(r, "vsE0ud_d_logloss_ud")])
    per = table(["หุ้น / รอบ", "โมเดล", "แม่น / log loss", "Δแม่น vs E0-ud [CI]", "Δlog loss vs E0-ud [CI]"],
                rows, "num")
    return main, per


def price_section(t):
    r = t[(t["model"] == "E9-LR")].set_index("group")
    b = t[(t["model"] == "E0 baseline (รอบแรก)")].set_index("group")
    d = pd.read_csv(R2 / "e9_price_detail.csv").set_index("group")
    rows = []
    for g in ["ALL"] + GROUPS:
        rows.append([GROUP_TH[g], num(d.at[g, "mae_point_baht"], 2), num(d.at[g, "mae_point_ticks"], 2),
                     num(d.at[g, "mean_abs_gap_part_baht"], 2), num(d.at[g, "mean_abs_window_part_baht"], 2),
                     num(d.at[g, "mae_shift1tick_baht"], 2),
                     f'{pct(r.at[g, "coverage50"])} / {pct(r.at[g, "coverage80"])}',
                     f'{pct(b.at[g, "coverage50"])} / {pct(b.at[g, "coverage80"])}',
                     f'{num(r.at[g, "width80_baht"], 2)} ({num(b.at[g, "width80_baht"], 2)})',
                     f'{r.at[g, "pinball"] * 1e4:.1f} ({b.at[g, "pinball"] * 1e4:.1f})'])
    return table(["หุ้น / รอบ", "MAE (บาท)", "MAE (ช่อง)", "|ส่วน gap|", "|ส่วนในช่วง|", "MAE เลื่อน 1 ช่อง",
                  "coverage 50/80 (E9)", "coverage 50/80 (E0)", "กว้าง 80% บาท E9 (E0)", "pinball ×10⁴ E9 (E0)"],
                 rows, "num small")


def shap_section():
    s = pd.read_csv(R2 / "e9_gbm_shap.csv", index_col=0)
    s = s.sort_values("afternoon", ascending=False)
    rows = [[f"<code>{esc(i)}</code>", num(r.get("morning"), 4), num(r.get("afternoon"), 4)] for i, r in s.iterrows()]
    hp = pd.read_csv(R2 / "e9_chosen_hparams.csv")
    hp["share"] = hp["n"] / hp.groupby(["param", "session"])["n"].transform("sum")
    hrows = []
    for (param, sess), g in hp.groupby(["param", "session"]):
        hrows.append([param, "เช้า" if sess == "morning" else "บ่าย",
                      ", ".join(f'{v:g}: {100 * s:.0f}%' for v, s in zip(g["value"], g["share"]))])
    return (table(["feature", "เช้า", "บ่าย"], rows, "num"),
            table(["ค่า", "รอบ", "ค่าที่ถูกเลือก (สัดส่วนของรอบทดสอบ)"], hrows))


def e10_section(t):
    e10 = t[(t["exp"] == "E10") & (t["group"] == "ALL")]
    if e10.empty:
        return "<p>ยังไม่มีผล E10</p>", "", "", ""
    times = variant_times()
    seeds = pd.read_csv(R2 / "seeds.csv").set_index("model") if (R2 / "seeds.csv").exists() else pd.DataFrame()
    vs = pd.read_csv(R2 / "e10_vs_e9_e7.csv") if (R2 / "e10_vs_e9_e7.csv").exists() else pd.DataFrame()
    vs = vs[vs["group"] == "ALL"].set_index("model") if len(vs) else vs
    rows = []
    for _, r in e10.iterrows():
        arch, st = r["variant"].split("_s", 1)
        st = "S" + st.replace("_x20", " (E10-20)")
        sd = seeds.loc[r["model"]] if r["model"] in seeds.index else None
        seed_txt = "–" if sd is None else f'{pct(sd["acc_nf_min"])}–{pct(sd["acc_nf_max"])}'
        v = vs.loc[r["model"]] if len(vs) and r["model"] in vs.index else {}
        rows.append([ARCH_TH.get(arch, arch), st, pct(r["acc_nf"]), seed_txt, num(r["logloss_ud"], 4),
                     ci_pp(r, "vsE0ud_d_acc_nf"),
                     ci_pp(v, "vsbest_E9_d_acc_nf") if len(v) else "–",
                     ci_pp(v, "vsE7a_s1_d_acc_nf") if len(v) else "–",
                     f'{times[r["variant"]] / 60:.0f}' if r["variant"] in times else "–"])
    main = table(["โมเดล", "ขั้น", "แม่น (ไม่นับ flat)", "ช่วงราย seed", "log loss ขึ้น/ลง", "Δแม่น vs E0-ud",
                  "Δแม่น vs E9 ดีสุด", "Δแม่น vs E7a ขั้น 1", "นาที"], rows, "num small")
    curves = []
    for _, r in e10.iterrows():
        f = EXP / "E10" / f"{r['variant']}_curves.csv"
        if f.exists() and f.stat().st_size > 10:   # S0 ไม่มี epoch → ไฟล์ว่าง
            c = pd.read_csv(f)
            if len(c):
                arch, st = r["variant"].split("_s", 1)
                curves.append(f'<div class="cell">{curve_svg(c, f"{ARCH_TH.get(arch, arch)} S{st}")}</div>')
    curve_html = ('<div class="grid3">' + "".join(curves) + "</div>") if curves else ""
    forest = forest_svg([(f'{ARCH_TH.get(r["variant"].split("_s")[0])} S{r["variant"].split("_s", 1)[1]}',
                          r["vsE0ud_d_acc_nf"], r["vsE0ud_d_acc_nf_lo"], r["vsE0ud_d_acc_nf_hi"])
                         for _, r in e10.iterrows() if pd.notna(r["vsE0ud_d_acc_nf"])])
    best9 = vs["best_E9"].iat[0] if len(vs) else "–"
    return main, curve_html, forest, best9


def zoo_section():
    s = EXP / "EZOO" / "summary.csv"
    b = EXP / "EZOO" / "bootstrap.csv"
    if not s.exists():
        return "<p>ยังไม่มีผล</p>"
    s = pd.read_csv(s)
    b = pd.read_csv(b) if b.exists() else pd.DataFrame()
    rows = []
    for _, r in s[s["group"] == "ALL"].iterrows():
        bb = b[(b["comparison"] == f'{r["variant"]} − baseline_ud') & (b["metric"] == "err_nf")]
        ci = "–" if bb.empty else (f'{-100 * bb["mean_diff"].iat[0]:+.1f} [{-100 * bb["ci_high"].iat[0]:+.1f}, '
                                   f'{-100 * bb["ci_low"].iat[0]:+.1f}]')
        rows.append([esc(r["variant"]), pct(r["acc_flat_excluded"]), num(r["logloss_ud"], 4), ci])
    return table(["โมเดล (ค่า default)", "แม่น (ไม่นับ flat)", "log loss ขึ้น/ลง", "Δแม่น vs E0-ud [CI]"], rows, "num")


def calib_svg(c, title, width=230, height=200):
    """calibration: ค่าเฉลี่ยที่ทาย (แกนนอน) vs สัดส่วนจริง + 95% CI (แกนตั้ง) ต่อ 5 ช่วง; เส้นประ = ตรงกันพอดี"""
    l, r, t, b = 34, 8, 20, 26
    lo, hi = 0.0, 0.45
    X = lambda v: l + (v - lo) / (hi - lo) * (width - l - r)
    Y = lambda v: t + (hi - v) / (hi - lo) * (height - t - b)
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{esc(title)}">',
           f'<text x="{l}" y="12" class="lab">{esc(title)}</text>']
    for v in np.arange(0, hi + 1e-9, 0.1):
        out.append(f'<line x1="{l}" x2="{width - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{l - 4}" y="{Y(v) + 3:.1f}" class="tick" text-anchor="end">{v:.1f}</text>'
                   f'<text x="{X(v):.1f}" y="{height - 10}" class="tick" text-anchor="middle">{v:.1f}</text>')
    out.append(f'<line x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" class="zero"/>')
    for _, q in c.iterrows():
        x = X(q["mean_pred"])
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(min(q["obs_hi"], hi)):.1f}" y2="{Y(q["obs_lo"]):.1f}" class="ci"/>'
                   f'<circle cx="{x:.1f}" cy="{Y(q["observed"]):.1f}" r="3.5" class="dot"/>')
    out.append("</svg>")
    return "".join(out)


def flat_section():
    """E9-FLAT: Model A, calibration, 3 คำตอบ, กฎราคาค่ามัธยฐาน (outputs/reports/round2/flat_*.csv)"""
    if not (R2 / "flat_model_a.csv").exists():
        return {k: "<p>ยังไม่มีผล</p>" for k in ("a", "calib", "three", "price")}
    a = pd.read_csv(R2 / "flat_model_a.csv")
    rows = []
    for _, r in a[a["group"] == "ALL"].iterrows():
        rows.append([r["target"], r["model"].upper() if r["model"] != "base" else "baseline (สัดส่วนในอดีต)",
                     pct(r["flat_rate"]), num(r["mean_p"], 3), num(r["logloss"], 4), num(r["brier"], 4),
                     ci_ll(r, "d_logloss"), ci_ll(r, "d_brier")])
    a_all = table(["เป้าหมาย", "โมเดล", "นิ่งจริง", "P(นิ่ง) เฉลี่ย", "log loss", "Brier", "Δlog loss vs baseline",
                   "ΔBrier vs baseline"], rows, "num small")
    rows = []
    for g in GROUPS:
        for tg in ("A-dir", "A-price"):
            for m in ("lr", "gbm"):
                r = a[(a["group"] == g) & (a["target"] == tg) & (a["model"] == m)].iloc[0]
                rows.append([GROUP_TH[g] if (tg, m) == ("A-dir", "lr") else "", tg, m.upper(), pct(r["flat_rate"]),
                             ci_ll(r, "d_logloss"), ci_ll(r, "d_brier")])
    a_grp = table(["หุ้น / รอบ", "เป้าหมาย", "โมเดล", "นิ่งจริง", "Δlog loss vs baseline", "ΔBrier vs baseline"],
                  rows, "num small")
    c = pd.read_csv(R2 / "flat_calibration.csv")
    cells = []
    for tg in ("A-dir", "A-price"):
        for g in GROUPS:
            cc = c[(c["target"] == tg) & (c["model"] == "lr") & (c["group"] == g)]
            cells.append(f'<div class="cell">{calib_svg(cc, f"{tg} LR · {GROUP_TH[g]}")}</div>')
    calib = '<div class="grid4">' + "".join(cells) + "</div>"
    th = pd.read_csv(R2 / "flat_three_way.csv")
    rows = []
    for _, r in th[th["group"] == "ALL"].iterrows():
        d = "—" if pd.isna(r.get("d_lo")) else f'{pp(r["d_main_minus_this"])} [{pp(r["d_lo"])}, {pp(r["d_hi"])}]' + (
            " ✱" if r["d_lo"] > 0 or r["d_hi"] < 0 else "")
        rows.append([esc(r["answer"]), pct(r["acc3"]), pct(r["share_up"]), pct(r["share_flat"]), pct(r["share_down"]), d])
    three = table(["คำตอบ", "แม่น (นับ \"เท่าเดิม\" ถูกได้)", "ทายขึ้น", "ทายเท่าเดิม", "ทายลง",
                   "แถวแรก − แถวนี้ [CI]"], rows, "num small")
    pr = pd.read_csv(R2 / "flat_price_rule.csv")
    rows = []
    for _, r in pr.iterrows():
        rows.append([GROUP_TH[r["group"]], f'{r["n_shift_up"]} / {r["n_shift_down"]} จาก {r["n"]}',
                     f'{num(r["mae_z0_baht"], 3)} / {num(r["mae_rule_baht"], 3)}',
                     f'{num(r["mae_z0_ticks"], 3)} / {num(r["mae_rule_ticks"], 3)}',
                     ci_ll(r.rename({"d_mae_baht": "d", "d_mae_lo": "d_lo", "d_mae_hi": "d_hi"}), "d")])
    price = table(["หุ้น / รอบ", "เลื่อน +1 / −1 ช่อง", "MAE บาท (z=0 / กฎ)", "MAE ช่อง (z=0 / กฎ)",
                   "ΔMAE บาท [CI]"], rows, "num small")
    return {"a": a_all + a_grp, "calib": calib, "three": three, "price": price}


CSS = """
:root { --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e; --rule:#e2e1dc;
        --series-1:#2a78d6; --series-2:#c2410c; --accent-bg:#f3f6fb; --warn-bg:#fff6e5; }
@page { size: A4; margin: 15mm 14mm 15mm 14mm; }
body { font-family: "Sarabun", "Leelawadee UI", "Tahoma", sans-serif; color: var(--text-primary);
       background: #fff; font-size: 10.3pt; line-height: 1.55; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 2mm; }
h2 { font-size: 13.5pt; margin: 6mm 0 2mm; border-bottom: 1.5px solid var(--text-primary); padding-bottom: 1mm; break-after: avoid; }
h3 { font-size: 11.2pt; margin: 4mm 0 1.5mm; break-after: avoid; }
.sub { color: var(--text-secondary); margin-bottom: 4mm; }
table { border-collapse: collapse; width: 100%; margin: 1.5mm 0 2.5mm; font-size: 9pt; break-inside: avoid; }
table.small { font-size: 8.2pt; }
th { text-align: left; font-weight: 600; color: var(--text-secondary); border-bottom: 1px solid var(--text-primary);
     padding: 1.1mm 1.3mm; vertical-align: bottom; }
td { border-bottom: 1px solid var(--rule); padding: 1mm 1.3mm; vertical-align: top; }
table.num td:not(:first-child) { font-variant-numeric: tabular-nums; white-space: nowrap; }
code { font-family: Consolas, monospace; font-size: 8.6pt; }
.box { background: var(--accent-bg); border-left: 3px solid var(--series-1); padding: 2.5mm 4mm; margin: 2mm 0 3mm; break-inside: avoid; }
.warn { background: var(--warn-bg); padding: 1.5mm 3mm; }
.note { color: var(--text-secondary); font-size: 8.8pt; }
ul { margin: 1mm 0 2mm; padding-left: 6mm; } li { margin: 0.6mm 0; }
figure { margin: 2mm 0 3mm; break-inside: avoid; }
figcaption { font-size: 9.3pt; color: var(--text-secondary); margin-bottom: 1mm; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2mm 4mm; break-inside: avoid; }
.grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2mm 3mm; break-inside: avoid; }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .zero { stroke: var(--text-secondary); stroke-width: 1.5; stroke-dasharray: 4 3; }
svg .ci { stroke: var(--series-1); stroke-width: 2; stroke-linecap: round; }
svg .dot { fill: var(--series-1); stroke: #fff; stroke-width: 2; }
svg .line { fill: none; stroke-width: 2; } svg .s1 { stroke: var(--series-1); } svg .s2 { stroke: var(--series-2); }
svg text { font-family: inherit; } svg .lab { font-size: 11px; fill: var(--text-primary); }
svg .tick, svg .val { font-size: 9.5px; fill: var(--text-secondary); }
.key span { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin: 0 4px 0 10px; }
.pb { break-before: page; }
"""


def build_html(narrative):
    t = metrics()
    e9_main, e9_per = e9_section(t)
    shap, hp = shap_section()
    e10_main, curves, forest10, best9 = e10_section(t)
    fl = flat_section()
    forest9 = forest_svg([(r["model"], r["vsE0ud_d_acc_nf"], r["vsE0ud_d_acc_nf_lo"], r["vsE0ud_d_acc_nf_hi"])
                          for _, r in t[(t["group"] == "ALL") & t["vsE0ud_d_acc_nf"].notna()].iterrows()
                          if r["exp"] != "E10"])
    n = t[(t["group"] == "ALL")]["n_nonflat"].iat[0]
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><title>รายงานรอบที่ 2</title>
<style>{CSS}</style></head><body>
<h1>ผลการทดสอบรอบที่ 2: ตัวชี้วัดใหม่, E9, E9-FLAT และ E10</h1>
<div class="sub">KBANK / ADVANC รอบเช้า (10:00→12:30) และบ่าย (14:00→16:30) · walk-forward
24/04/2024 – 12/10/2026 ({n:,} รอบที่ไม่ flat) · holdout 10 วันสุดท้ายยังไม่แตะ · สร้างเมื่อ {datetime.now():%d/%m/%Y %H:%M}</div>
{narrative["summary"]}

<h2>0. ข้อมูลที่ใช้และวิธีเทรน</h2>
{narrative["data"]}

<h2>1. ตัวชี้วัดใหม่</h2>
<p>อาจารย์ให้คะแนนการทาย "ขึ้นหรือลง" วันที่ราคาจบที่เดิม (flat) จึงไม่มีคำตอบที่ถูก ตัวชี้วัดหลักรอบนี้คือ
<b>ความแม่นแบบไม่นับ flat</b>, ความแม่นแบบ flat = ผิด และ <b>log loss ขึ้น vs ลง</b> (เฉพาะวันไม่ flat)
เทียบกับ baseline ด้วย block bootstrap รายวัน 2,000 ครั้ง (✱ = CI ไม่คร่อม 0)</p>
<p><b>baseline มีสองแบบ:</b> E0 รอบแรกให้ P(ขึ้น) = สัดส่วน "ขึ้น" ของทุกวัน (ราว 0.38 จึงทาย "ลง" เกือบทุกครั้ง)
ส่วน <b>E0-ud</b> ใช้สัดส่วนขึ้นในวันที่ไม่ flat ซึ่งตรงกับเป้าหมายใหม่ จึงใช้ E0-ud เป็นตัวเทียบหลัก</p>

<h3>1.1 ผลรอบแรกเมื่อคิดด้วยตัวชี้วัดใหม่ (รวมทุกหุ้น/รอบ)</h3>
{round1_section(t)}
<p class="note">โมเดลรอบแรกฝึกเป้าหมาย "ขึ้น vs ไม่ขึ้น" ตัวเลขนี้คิดจาก P(ขึ้น) เดิมโดยไม่ปรับ
log loss ขึ้น/ลงของโมเดลรอบแรกจึงเสียเปรียบ เพราะ P(ขึ้น) ต่ำกว่าสัดส่วนจริงในวันไม่ flat</p>
{narrative["round1"]}

<h3>1.2 สัดส่วน "ขึ้น" ในวันไม่ flat รายไตรมาส (%)</h3>
{quarter_section()}
<p class="note">ไตรมาสที่มีน้อยกว่า 30 รอบไม่แสดง (2023 Q4 เช้า, 2024 Q1 บ่าย และ 2026 Q4)</p>
{narrative["quarter"]}

<h3>1.3 feature "ผลตอบแทน 30 นาทีสุดท้าย"</h3>
{narrative["last30"]}

<h2 class="pb">2. E9: โมเดลตารางรอบใหม่</h2>
{narrative["e9_method"]}
<h3>2.1 ทิศทาง (รวมทุกหุ้น/รอบ)</h3>
{e9_main}
<figure><figcaption>ผลต่างความแม่น (ไม่นับ flat) เทียบ E0-ud และ 95% CI (จุด %; ขวาของเส้นประ = ดีกว่า)</figcaption>
{forest9}</figure>
{narrative["e9_result"]}
<h3>2.2 แยกหุ้น / รอบ</h3>
{e9_per}
<h3>2.3 SHAP ของ E9-GBM (ค่าเฉลี่ย |SHAP| บนรอบทดสอบ) และค่าที่ time-series CV ภายในเลือก</h3>
<div class="grid3" style="grid-template-columns: 1fr 1.4fr">{shap}{hp}</div>
{narrative["shap"]}
<h3>2.4 ราคา (ราคาจุด = ราคาล่าสุด ณ จุดตัด, ช่วงราคา conformal × σ̂)</h3>
{price_section(t)}
{narrative["price"]}

<h2 class="pb">3. E9-FLAT: ทายวันนิ่งล่วงหน้า</h2>
{narrative["flat_method"]}
<h3>3.1 Model A: โอกาสนิ่ง (รวมทุกหุ้น/รอบ และแยกหุ้น/รอบ)</h3>
{fl["a"]}
<h3>3.2 Calibration ของ LR (5 ช่วงตาม quantile ของ P(นิ่ง) ที่ทาย; เส้นตั้ง = 95% CI ของสัดส่วนจริง)</h3>
{fl["calib"]}
{narrative["flat_a"]}
<h3>3.3 ทิศทางแบบ 3 คำตอบ</h3>
{fl["three"]}
{narrative["flat_three"]}
<h3>3.4 ราคาแบบกฎค่ามัธยฐาน</h3>
{fl["price"]}
{narrative["flat_price"]}

<h2 class="pb">4. E10: CNN pretrained แบบ fine-tune</h2>
{narrative["e10_method"]}
{e10_main}
<p class="note">ทุกโมเดลใช้ภาพ scalogram, pooled 2 หุ้น (ยกเว้นแถว E10-20), refit ทุก 20 วันทำการ วันทดสอบชุดเดียวกับ E9
"ช่วงราย seed" = ความแม่นของแต่ละ seed (3 seed) ส่วนตัวเลขหลักมาจากการเฉลี่ยความน่าจะเป็น 3 seed;
E9 ดีสุด = {esc(best9)}; ✱ = CI ไม่คร่อม 0; "นาที" = เวลาฝึกทั้ง walk-forward บน RTX 5070 Laptop</p>
<figure><figcaption>ผลต่างความแม่น (ไม่นับ flat) ของ E10 เทียบ E0-ud และ 95% CI (จุด %)</figcaption>{forest10}</figure>
{narrative["e10_result"]}
<h3>4.1 Learning curve</h3>
<p class="note">ค่าเฉลี่ย loss ต่อ epoch ของทุก fold และ seed <span class="key"><span style="background:#c2410c"></span>train
(label smoothing 0.1)<span style="background:#2a78d6"></span>validation (20% วันท้ายของชุดฝึก)</span>
— แสดงเฉพาะ epoch ที่ยังมี fold × seed ฝึกอยู่อย่างน้อยครึ่งหนึ่ง (ที่เหลือ early stop ไปแล้ว)</p>
{curves}
{narrative["curves"]}

<h2>5. ภาคผนวก: model zoo (ค่า default ไม่จูน ไม่ใช้เลือกโมเดล)</h2>
{zoo_section()}
{narrative["zoo"]}

<h2 class="pb">6. เลือกโมเดลส่งจริง</h2>
{narrative["selection"]}

<h2>7. วัดความหน่วงและระบบวันจริง</h2>
{narrative["live"]}

<h2>8. ยังต้องยืนยันกับอาจารย์</h2>
{narrative["questions"]}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "reports" / "report_round2.pdf")
    ap.add_argument("--narrative", type=Path, default=OUTPUT_DIR / "reports" / "round2" / "narrative.py",
                    help="ไฟล์ข้อความบรรยาย (dict NARRATIVE) — แยกจากตัวเลขเพื่อแก้ได้โดยไม่ยุ่งกับโค้ด")
    args = ap.parse_args()
    load_rules()
    ns = {}
    exec(args.narrative.read_text(encoding="utf-8"), ns)
    src = args.out.with_suffix(".html")
    src.write_text(build_html(ns["NARRATIVE"]), encoding="utf-8")
    browser = r1.find_browser()
    if not browser:
        raise SystemExit(f"ไม่พบ Edge/Chrome — เปิด {src} แล้วสั่งพิมพ์เป็น PDF เอง")
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
