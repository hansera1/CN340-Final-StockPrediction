"""E27: วิเคราะห์ pattern ของการทาย gap — ดู results/e27_preregistration.md · รายงานเท่านั้น

    python experiments/54_e27_gap_analysis.py fetch   # ดึงข้อมูลจาก Yahoo เก็บใน data/raw/external/
    python experiments/54_e27_gap_analysis.py run
→ results/e27/ (ตัวเลขสรุปเท่านั้น)
"""
import importlib.util
import sys
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd

from forecast.config import EXTERNAL_DIR, OUTPUT_DIR, RAW_DIR, ROOT, load_rules
from forecast.features import overnight as ov

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
TICKERS = {"Z74.SI": "Z74_SI", "D05.SI": "D05_SI", "^BKX": "BKX", "TLT": "TLT", "^TNX": "TNX"}
OPEN_TIME = {"Z74.SI": ("09:00", "Asia/Singapore"), "D05.SI": ("09:00", "Asia/Singapore")}
CLOSE_TIME = {"^BKX": ("16:00", "America/New_York"), "TLT": ("16:00", "America/New_York"), "^TNX": ("16:00", "America/New_York")}
NEW = ["fx_xd", "fx_singtel_gap", "fx_dbs_gap", "fx_bkx", "fx_tlt", "fx_tnx"]
EXPECTED = {"ADVANC": ["fx_xd", "fx_singtel_gap", "fx_tlt", "fx_tnx"], "KBANK": ["fx_xd", "fx_dbs_gap", "fx_bkx", "fx_tlt", "fx_tnx"]}
RES, EXT = ROOT / "results" / "e27", RAW_DIR / "external"
ov.OPEN_TIME.update(OPEN_TIME)
ov.CLOSE_TIME.update(CLOSE_TIME)


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fetch():
    import yfinance as yf
    EXT.mkdir(parents=True, exist_ok=True)
    for t, n in TICKERS.items():
        df = yf.download(t, start="2023-06-01", end="2026-11-07", interval="1d", auto_adjust=False, progress=False, multi_level_index=False)
        if df is None or df.empty:
            print(t, "ไม่มีข้อมูล", flush=True)
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(EXT / f"{n}_1d.csv")
        print(t, len(df), df.index.min().date(), df.index.max().date(), "open_na", int(df["Open"].isna().sum()), flush=True)


def load_new():
    out = {}
    for t, n in TICKERS.items():
        p = EXT / f"{n}_1d.csv"
        if p.exists():
            out[t] = pd.read_csv(p, index_col=0, parse_dates=True)
    return out


def level_change(s, a, b):
    return ov.asof(s, b) - ov.asof(s, a)


def build_new(days, tables):
    """feature ต่อวันไทย · หน้าต่างข้ามคืนเดียวกับ E18 = (16:30 วันทำการก่อน, 08:30 วันนี้] · ใช้เฉพาะค่าที่พร้อมก่อน 08:30"""
    rows = []
    close = {t: ov.close_series(tables[t], t) for t in CLOSE_TIME if t in tables}
    for i in range(1, len(days)):
        d, prev = days[i], days[i - 1]
        a, b = prev.normalize() + pd.Timedelta("16:30:00"), d.normalize() + pd.Timedelta("08:30:00")
        r = {"date": d}
        r["fx_singtel_gap"] = ov.gap(tables["Z74.SI"], "Z74.SI", d, b) if "Z74.SI" in tables else np.nan
        r["fx_dbs_gap"] = ov.gap(tables["D05.SI"], "D05.SI", d, b) if "D05.SI" in tables else np.nan
        r["fx_bkx"] = ov.window_change(close["^BKX"], a, b) if "^BKX" in close else np.nan
        r["fx_tlt"] = ov.window_change(close["TLT"], a, b) if "TLT" in close else np.nan
        r["fx_tnx"] = level_change(close["^TNX"], a, b) if "^TNX" in close else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def xd_feature(am, calendar):
    """ผลทางกลของเงินปันผล: ln((P(c) − ปันผล) / P(c)) วัน XD ของหุ้นนั้น · วันอื่น = 0"""
    x = calendar[calendar["event"] == "XD"].copy()
    x["_day"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
    div = x.groupby(["symbol", "_day"])["amount"].sum()
    day = am["date"].astype(str).str[:10]
    amt = np.array([div.get((s, k), 0.0) for s, k in zip(am["symbol"], day)], float)
    return np.log((am["p_ref"].to_numpy(float) - amt) / am["p_ref"].to_numpy(float)), amt


def gap_with(am, cols, s38):
    """walk-forward แบบ E18b โดยเปลี่ยนชุด feature → แถว KEYS + g_hat"""
    keep = s38.FX
    s38.FX = cols
    try:
        gh, _ = s38.gap_oos(am)
    finally:
        s38.FX = keep
    s38.assert_no_leak(gh)
    return gh[KEYS + ["g_hat"]]


def price_table(x, n_boot, s38, s31):
    pe, dates, rows = x["p_end"].to_numpy(float), x["date"].astype(str).to_numpy(), []
    base = pe - x["pt_PR-A"].to_numpy(float)
    groups = [("เช้า", np.ones(len(x), bool))] + [(f"{s} เช้า", (x["symbol"] == s).to_numpy()) for s in ("KBANK", "ADVANC")]
    for name in ["PR-A", "H2b-rule", "Oracle-rule", "P(10:00)"]:
        e = pe - x[f"pt_{name}"].to_numpy(float)
        for lab, m in groups:
            r = {"model": name, "group": lab, "n": int(m.sum()), "mae": np.abs(e[m]).mean(), "rmse": np.sqrt((e[m] ** 2).mean())}
            if name != "PR-A":
                r.update(zip(["d_mae", "d_mae_lo", "d_mae_hi"], s31.boot_ci(np.abs(e[m]) - np.abs(base[m]), dates[m], 0.95, n_boot)))
                r.update(zip(["d_rmse", "d_rmse_lo", "d_rmse_hi"], s38.boot_rmse(e[m], base[m], dates[m], 0.95, n_boot)))
            rows.append(r)
    return pd.DataFrame(rows)


def run():
    s48, s18 = _module("48_e23_skip_fx.py"), _module("18_score_round4.py")
    s37 = _module("37_e18_overnight.py")
    rules = load_rules()
    tz, n_boot = rules["timezone"], rules["evaluation"]["n_boot"]
    ds, exps, s31, _, s38, make_fx = s48.prepare(rules)
    RES.mkdir(parents=True, exist_ok=True)

    # ส่วน 1
    x = pd.read_csv(OUTPUT_DIR / "experiments" / "E18b" / "points.csv")
    x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_convert(tz)
    assert len(x) == 1196 and x["p_start"].notna().all()
    x["pt_Oracle-rule"] = s18.price_rule(x["p_start"].to_numpy(float), x["eq_A"].to_numpy(float), x["up_A"].to_numpy(float))[0]
    x["pt_P(10:00)"] = x["p_start"].to_numpy(float)
    t = price_table(x, n_boot, s38, s31)
    t.to_csv(RES / "oracle.csv", index=False, encoding="utf-8-sig")

    # ส่วน 2
    fx = make_fx(ds)
    am, _ = s38.morning_frame(ds, fx)
    tables = load_new()
    new = build_new(pd.DatetimeIndex(np.sort(ds["date"].unique())), tables)
    am = am.merge(new, on="date", how="left")
    cal = pd.read_csv(EXTERNAL_DIR / "calendar.csv")
    am["fx_xd"], am["_div"] = xd_feature(am, cal)
    na = am[NEW].isna().sum().to_dict()
    am[NEW] = am[NEW].fillna(0.0)
    base_cols = list(s38.FX)
    sets = {"ชุดเดิม 4 ตัว": base_cols}
    for c in NEW:
        sets[f"{c} ตัวเดียว"] = [c]
        sets[f"ชุดเดิม + {c}"] = base_cols + [c]
    sets["ชุดเดิม + ทุกตัว"] = base_cols + NEW
    rows, wf = [], None
    for name, cols in sets.items():
        gh = gap_with(am, cols, s38)
        j = am.merge(gh, on=KEYS)
        for sym, g in j.groupby("symbol"):
            g = g[s38.wf_rows(g, rules)]
            r, lo, hi, n = s37.boot_corr(g["g_hat"].to_numpy(), g["gap"].to_numpy(), g["date"].to_numpy(), n_boot)
            feature = next((c for c in NEW if c in name), "")
            rows.append({"set": name, "symbol": sym, "n": n, "corr": r, "lo": lo, "hi": hi, "expected_pair": feature in EXPECTED[sym] if feature else True})
    c = pd.DataFrame(rows)
    c.to_csv(RES / "feature_corr.csv", index=False, encoding="utf-8-sig")

    days = am[s38.wf_rows(am, rules)]
    xd = days[days["_div"] > 0]
    xd_tab = pd.DataFrame([{"symbol": s, "xd_days": len(g), "mean_gap": g["gap"].mean(), "mean_fx_xd": g["fx_xd"].mean(), "mean_gap_minus_xd": (g["gap"] - g["fx_xd"]).mean()} for s, g in xd.groupby("symbol")])
    xd_tab.to_csv(RES / "xd_days.csv", index=False, encoding="utf-8-sig")
    xa = x.merge(am[KEYS + ["_div"]], on=KEYS)
    m = (xa["_div"] > 0).to_numpy()
    xd_price = []
    for lab, pts in [("PR-A", xa["pt_PR-A"].to_numpy(float)), ("P(c)", xa["pc"].to_numpy(float)), ("P(c) − ปันผล", xa["pc"].to_numpy(float) - xa["_div"].to_numpy(float))]:
        e = xa["p_end"].to_numpy(float) - pts
        xd_price.append({"model": lab, "n_xd_rows": int(m.sum()), "mae_xd": np.abs(e[m]).mean(), "rmse_xd": np.sqrt((e[m] ** 2).mean())})
    xp = pd.DataFrame(xd_price)
    xp.to_csv(RES / "xd_price.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(t.round(3).to_string(index=False))
    print("ค่าว่างที่แทน 0:", na)
    print(c.round(4).to_string(index=False))
    print(xd_tab.round(4).to_string(index=False))
    print(xp.round(3).to_string(index=False))
    print("gap รวม sd", days.groupby("symbol")["gap"].std().round(5).to_dict(), "เช้า sd หลังเปิด", np.log(days["p_end"] / days["p_start"]).groupby(days["symbol"]).std().round(5).to_dict())


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]]()
