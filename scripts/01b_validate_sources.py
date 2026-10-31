"""ตรวจว่าข้อมูล backtest (yfinance 60m) ตรงกับข้อมูลรายนาที ในวันที่ข้อมูลซ้อนกัน

    python scripts/01b_validate_sources.py
    → outputs/reports/validate_coarse_vs_1m.csv, validate_label_points.csv, validate_features.csv

0) แท่งหยาบ (Settrade 30m, yfinance 60m) เทียบ Settrade 1m รายวัน: open ประมูลเปิด, close ≤ 12:29 / ≤ 16:29,
   และ volume แท่งสุดท้ายต้องไม่รวมแท่งประมูลปิด (ใช้ volume ตัดสินเพราะบางวันราคา 16:29 = ATC)

1) ราคาจุด label: 10:00, 12:30, 14:00, 16:30 จากแท่ง 60m เทียบกับ Settrade 1m
   และราคา 16:30 จาก 60m ตรงกับ "ราคาซื้อขายล่าสุด ณ 16:30" หรือ "ราคาปิดประมูล (ATC)"
2) features รอบบ่าย: backtest (ไฟล์ 60m) เทียบกับเส้นทางวันจริง (1m วันนี้ → รวมเป็น 60m) ทีละแหล่ง
"""
import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import OUTPUT_DIR, load_rules, raw_source_dir, session_times
from forecast.data.io import load_bars
from forecast.features.dataset import assemble_dataset, feature_cols
from forecast.labels.make_labels import labels_for
from forecast.live.sources import live_bars

POINTS = {("morning", "p_start"): "10:00", ("morning", "p_end"): "12:30",
          ("afternoon", "p_start"): "14:00", ("afternoon", "p_end"): "16:30"}


def labels_from(rules, sym, source, interval):
    try:
        bars = load_bars(sym, rules, raw_source_dir(rules, source), interval)
    except FileNotFoundError:
        return None
    return labels_for(bars, {**rules, "label": {**rules["label"], "store_closing_auction": True}}, interval)


def label_points(rules):
    rows = []
    for sym in rules["symbols"]:
        ref = labels_from(rules, sym, "settrade", "1m")
        cands = {"yf60m": labels_from(rules, sym, "yfinance", rules["data"]["interval"]),
                 "yf1m": labels_from(rules, sym, "yfinance", "1m")}
        if ref is None:
            continue
        for _, r in ref.iterrows():
            for (sess, col), hhmm in POINTS.items():
                if r["session"] != sess:
                    continue
                row = {"date": r["date"].date(), "symbol": sym, "point": hhmm, "settrade_1m": r[col]}
                if hhmm == "16:30":
                    row["settrade_atc"] = r.get("p_close_auction", np.nan)
                for name, lab in cands.items():
                    hit = lab[(lab["date"] == r["date"]) & (lab["session"] == sess)] if lab is not None else []
                    row[name] = hit[col].iat[0] if len(hit) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def report_points(df):
    print("\n=== 1) ราคาจุด label: yfinance 60m เทียบ Settrade 1m ===")
    df = df[df["yf60m"].notna() & df["settrade_1m"].notna()]
    for point, g in df.groupby("point", sort=False):
        same = np.isclose(g["yf60m"], g["settrade_1m"])
        line = f"  {point}: ตรงกัน {same.sum()}/{len(g)}"
        if point == "16:30":
            atc = np.isclose(g["yf60m"], g["settrade_atc"])
            differ = ~np.isclose(g["settrade_1m"], g["settrade_atc"])   # วันที่สองนิยามให้ราคาต่างกัน
            line += (f"  | ตรงกับ ATC {atc.sum()}/{len(g)}"
                     f"  | วันที่ราคา 16:30 ≠ ATC: {differ.sum()} วัน → 60m ตรงกับ 16:30 {(same & differ).sum()},"
                     f" ตรงกับ ATC {(atc & differ).sum()}")
        print(line)
    bad = df[~np.isclose(df["yf60m"], df["settrade_1m"])]
    if len(bad):
        print("  จุดที่ไม่ตรง:\n" + bad.to_string(index=False))


def _raw(rules, source, sym, interval):
    path = raw_source_dir(rules, source) / f"{sym}_{interval}.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path, parse_dates=["timestamp"])
    d["day"], d["hm"] = d["timestamp"].dt.normalize(), d["timestamp"].dt.strftime("%H:%M")
    return d


def coarse_vs_1m(rules, coarse=(("settrade", "30m"), ("yfinance", "60m"))):
    """ข้อ 2 ของ NEXT_STEPS: แท่งหยาบเทียบ Settrade 1m รายวัน (timestamp = เวลาเริ่มแท่ง)"""
    rows = []
    for sym in rules["symbols"]:
        m1 = _raw(rules, "settrade", sym, "1m")
        for src, iv in coarse:
            c = _raw(rules, src, sym, iv)
            if m1 is None or c is None:
                continue
            for day in sorted(set(m1["day"]) & set(c["day"])):
                a, b = m1[m1["day"] == day], c[c["day"] == day]
                am1, pm1, atc1 = a[a["hm"] < "12:30"], a[(a["hm"] >= "12:30") & (a["hm"] < "16:30")], a[a["hm"] >= "16:30"]
                amc, pmc = b[b["hm"] < "12:30"], b[(b["hm"] >= "12:30") & (b["hm"] < "16:30")]
                if am1.empty or pm1.empty or amc.empty or pmc.empty:
                    continue
                last_c = pmc.iloc[-1]
                in_last = a[(a["timestamp"] >= last_c["timestamp"]) & (a["hm"] < "16:30")]["volume"].sum()
                checks = {
                    "open_am": (amc["open"].iat[0], am1["open"].iat[0]),          # ประมูลเปิดเช้า
                    "close_1229": (amc["close"].iat[-1], am1["close"].iat[-1]),
                    "open_pm": (pmc["open"].iat[0], pm1["open"].iat[0]),          # ประมูลเปิดบ่าย
                    "close_1629": (last_c["close"], pm1["close"].iat[-1]),
                    "vol_last_excl_atc": (last_c["volume"], in_last),
                }
                for name, (cv, ref) in checks.items():
                    row = {"source": f"{src} {iv}", "date": day.date(), "symbol": sym, "check": name,
                           "coarse": cv, "from_1m": ref, "diff": cv - ref, "match": bool(np.isclose(cv, ref))}
                    if name == "close_1629":
                        row["atc_1m"] = atc1["close"].iat[-1] if len(atc1) else np.nan
                    if name == "vol_last_excl_atc":
                        row["vol_incl_atc_1m"] = in_last + atc1["volume"].sum()
                        row["first_bar_1m"] = am1["hm"].iat[0]
                    rows.append(row)
    return pd.DataFrame(rows)


def report_coarse(df):
    print("\n=== 0) แท่งหยาบเทียบ Settrade 1m รายวัน (ตรง/ทั้งหมด) ===")
    if df.empty:
        print("  ไม่มีวันที่ซ้อนกัน")
        return
    for src, g in df.groupby("source"):
        print(f"  [{src}] วัน: {', '.join(str(d) for d in sorted(g['date'].unique()))}")
        for check, h in g.groupby("check", sort=False):
            line = f"    {check:18s} {h['match'].sum()}/{len(h)}"
            if check == "vol_last_excl_atc":
                incl = np.isclose(h["coarse"], h["vol_incl_atc_1m"]).sum()
                line += f"   (ตรงกับแบบรวม ATC {incl}/{len(h)})"
            if check == "close_1629":
                line += f"   (ตรงกับ ATC {np.isclose(h['coarse'], h['atc_1m']).sum()}/{len(h)})"
            bad = h[~h["match"]]
            if len(bad):
                line += "   ไม่ตรง: " + "; ".join(f"{r.symbol} {r.date} {r.coarse:g} vs {r.from_1m:g} (ต่าง {r.diff:+g})"
                                                for r in bad.itertuples())
            print(line)


def feature_check(rules):
    """เส้นทางวันจริงต่อแหล่ง เทียบแถว backtest เดียวกัน (เฉพาะวันที่ทุกแหล่งมีแท่ง 1m)"""
    none = OUTPUT_DIR / "_no_external"
    full = assemble_dataset(rules, ext_dir=none)
    cols = feature_cols(full)
    dates = None
    for src in rules["live"]["sources"]:
        for sym in rules["symbols"]:
            try:
                d = set(load_bars(sym, rules, raw_source_dir(rules, src), "1m")["date"])
            except FileNotFoundError:
                d = set()
            dates = d if dates is None else dates & d
    dates = sorted(set(full["date"]) & (dates or set()))
    rows = []
    for date in dates:
        cutoff = session_times(rules, "afternoon", date)["cutoff"]
        a = full[(full["date"] == date) & (full["session"] == "afternoon")].set_index("symbol")[cols]
        for src in rules["live"]["sources"]:
            r1 = {**rules, "live": {**rules["live"], "sources": [src]}}
            by = {}
            for sym in rules["symbols"]:
                b, _ = live_bars(sym, r1, date, cutoff)
                by[sym] = b[b["bar_end"] <= cutoff].reset_index(drop=True)
            live = assemble_dataset(r1, bars_by_symbol=by, as_of=cutoff, live_row=(date, "afternoon"), ext_dir=none)
            b = live[(live["date"] == date) & (live["session"] == "afternoon")].set_index("symbol")[cols]
            for sym in rules["symbols"]:
                for c in cols:
                    rows.append({"date": date.date(), "symbol": sym, "source_1m": src, "feature": c,
                                 "backtest_60m": a.at[sym, c], "live": b.at[sym, c]})
    return pd.DataFrame(rows)


def report_features(df):
    print("\n=== 2) features รอบบ่าย: backtest (60m) เทียบวันจริง (1m → 60m) ===")
    if df.empty:
        print("  ไม่มีวันที่ทุกแหล่งมีแท่ง 1m ซ้อนกับไฟล์ 60m")
        return
    df["abs_diff"] = (df["backtest_60m"] - df["live"]).abs()
    df["both_nan"] = df["backtest_60m"].isna() & df["live"].isna()
    print(f"  วันที่เทียบ: {sorted(df['date'].unique())}")
    for src, g in df.groupby("source_1m"):
        g = g[~g["both_nan"]]
        exact = (g["abs_diff"] < 1e-9).mean()
        worst = g.groupby("feature")["abs_diff"].max().sort_values(ascending=False)
        worst = worst[worst > 1e-9]
        print(f"  [{src}] ค่าตรงกัน {exact:.1%} ของ {len(g)} ค่า; feature ที่ต่าง (ผลต่างสูงสุด):")
        print("    " + (", ".join(f"{k}={v:.4g}" for k, v in worst.head(12).items()) or "ไม่มี"))


def main():
    rules = load_rules()
    out = OUTPUT_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    cv = coarse_vs_1m(rules)
    cv.to_csv(out / "validate_coarse_vs_1m.csv", index=False)
    report_coarse(cv)
    pts = label_points(rules)
    pts.to_csv(out / "validate_label_points.csv", index=False)
    report_points(pts)
    feats = feature_check(rules)
    feats.to_csv(out / "validate_features.csv", index=False)
    report_features(feats)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
