"""การประเมินทางการเงิน: XIRR และ Sharpe ratio ของสัญญาณทิศทางที่ใช้ส่งจริง — วิธีใน results/financial_preregistration.md

    python experiments/61_financial_eval.py
→ results/financial/ (ตัวเลขสรุปและ equity curve ไม่มีราคาดิบ) · การจำลองย้อนหลัง ไม่ใช่คำแนะนำการลงทุน
"""
import copy
import importlib.util
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from forecast.config import OUTPUT_DIR, RAW_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.runner import load_or_build_dataset, run_variant

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
NOTIONAL = 100_000.0
FEE = 0.00157 * 1.07          # 0.157% + VAT 7% ต่อขา
RF_YEAR = 0.01                # อัตราดอกเบี้ยนโยบาย ธปท. 1.00% (2569)
N_BOOT = 2000
STRATS = ["S1", "S2", "B1", "B3"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
RES = ROOT / "results" / "financial"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def trade_pnl(ret, pos, fee):
    """กำไรขาดทุนต่อรอบของเงินลงทุน NOTIONAL · pos = 1 ซื้อ, −1 ขายชอร์ต, 0 ถือเงินสด · ค่าธรรมเนียมขาเปิดบนมูลค่าเริ่ม ขาปิดบนมูลค่าตอนปิด"""
    ret, pos = np.asarray(ret, float), np.asarray(pos, float)
    gross = pos * ret * NOTIONAL
    cost = np.abs(pos) * fee * NOTIONAL * (2 + ret)
    return gross - cost


def positions(up, strategy):
    up = np.asarray(up, bool)
    return {"S1": up.astype(float), "S2": np.where(up, 1.0, -1.0), "B1": np.ones(len(up)), "B3": np.zeros(len(up))}[strategy]


def xirr(dates, flows):
    """อัตราผลตอบแทนต่อปี: Σ CFᵢ / (1 + r)^((dᵢ − d₀)/365) = 0"""
    d = pd.to_datetime(pd.Series(dates)).dt.tz_localize(None)
    t = ((d - d.min()).dt.days / 365.0).to_numpy()
    cf = np.asarray(flows, float)
    if np.allclose(cf.sum(), 0) and np.allclose(cf[1:-1], 0):
        return 0.0
    f = lambda r: np.sum(cf / (1 + r) ** t)
    try:
        return brentq(f, -0.9999, 1e6, maxiter=500)
    except ValueError:
        return np.nan


def sharpe(daily, rf_year=RF_YEAR):
    daily = np.asarray(daily, float)
    sd = daily.std(ddof=1)
    return np.nan if len(daily) < 2 or sd == 0 else (daily.mean() - rf_year / 252) / sd * np.sqrt(252)


def max_drawdown(equity):
    e = np.asarray(equity, float)
    peak = np.maximum.accumulate(e)
    return float(((e - peak) / peak).min())


def boot_sharpe(a, b=None, n=N_BOOT, seed=0):
    """CI 95% ของ Sharpe (และผลต่าง a − b) จาก bootstrap รายวัน"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    sa = np.array([sharpe(a[i]) for i in idx])
    out = (np.nanquantile(sa, .025), np.nanquantile(sa, .975))
    if b is None:
        return out
    sd = np.array([sharpe(a[i]) - sharpe(b[i]) for i in idx])
    return out, (sharpe(a) - sharpe(b), np.nanquantile(sd, .025), np.nanquantile(sd, .975))


def daily_series(rows, strategy, fee, days):
    """rows: ret, up, date ต่อช่อง → กำไรขาดทุนรายวัน (รวมทุกช่อง) บนทุกวันใน days"""
    pnl = trade_pnl(rows["ret"], positions(rows["up"], strategy), fee)
    s = pd.Series(pnl, index=rows["date"].to_numpy()).groupby(level=0).sum()
    return s.reindex(days, fill_value=0.0), pnl


def summarize(rows, period, label, fee, k_slots, b2=None):
    days = np.sort(rows["date"].unique())
    K = NOTIONAL * k_slots
    out, series = [], {}
    for st in STRATS:
        s, pnl = daily_series(rows, st, fee, days)
        series[st] = s / K
        traded = positions(rows["up"], st) != 0
        flows = np.r_[-K, s.to_numpy(), K]
        fdates = np.r_[days[:1], days, days[-1:]]
        eq = K + s.cumsum().to_numpy()
        out.append({"ช่วง": period, "รอบ": label, "กลยุทธ์": st, "ค่าธรรมเนียม": f"{fee:.4%}", "วัน": len(days), "เงินลงทุน": K,
                    "XIRR": xirr(fdates, flows), "Sharpe": sharpe(series[st].to_numpy()), "Sharpe_rf0": sharpe(series[st].to_numpy(), 0.0),
                    "ผลตอบแทนสะสม": s.sum() / K, "max_drawdown": max_drawdown(np.r_[K, eq]), "ครั้งที่ซื้อขาย": int(traded.sum()),
                    "อัตราชนะ": float((pnl[traded] > 0).mean()) if traded.any() else np.nan})
    if b2 is not None:
        s = b2.reindex(days, fill_value=0.0)
        Kb = NOTIONAL * 2
        series["B2"] = s / Kb
        eq = Kb + s.cumsum().to_numpy()
        out.append({"ช่วง": period, "รอบ": label, "กลยุทธ์": "B2", "ค่าธรรมเนียม": f"{fee:.4%}", "วัน": len(days), "เงินลงทุน": Kb,
                    "XIRR": xirr(np.r_[days[:1], days, days[-1:]], np.r_[-Kb, s.to_numpy(), Kb]), "Sharpe": sharpe(series["B2"].to_numpy()),
                    "Sharpe_rf0": sharpe(series["B2"].to_numpy(), 0.0), "ผลตอบแทนสะสม": s.sum() / Kb, "max_drawdown": max_drawdown(np.r_[Kb, eq]),
                    "ครั้งที่ซื้อขาย": 2, "อัตราชนะ": np.nan})
    t = pd.DataFrame(out)
    for st in series:
        lo, hi = boot_sharpe(series[st].to_numpy()) if len(days) > 2 else (np.nan, np.nan)
        t.loc[t["กลยุทธ์"] == st, ["Sharpe_lo", "Sharpe_hi"]] = [lo, hi]
    diff = None
    if len(days) > 2:
        _, (d, lo, hi) = boot_sharpe(series["S1"].to_numpy(), series["B1"].to_numpy())
        diff = {"ช่วง": period, "รอบ": label, "ค่าธรรมเนียม": f"{fee:.4%}", "n_วัน": len(days), "Sharpe_S1_ลบ_B1": d, "lo": lo, "hi": hi, "มีนัยสำคัญ": bool(lo > 0 or hi < 0)}
    return t, diff, series


def buy_hold(days, fee=0.0):
    """B2: กำไรขาดทุนรายวันของการถือ 2 หุ้น หุ้นละ NOTIONAL ราคาปิดทางการ Settrade 1d · ค่าธรรมเนียมขาซื้อวันแรก ขาขายวันสุดท้าย"""
    tot = None
    for sym in ("KBANK", "ADVANC"):
        d = pd.read_csv(RAW_DIR / "settrade" / f"{sym}_1d.csv", parse_dates=["timestamp"])
        c = pd.Series(d["close"].to_numpy(float), index=pd.DatetimeIndex(d["timestamp"]).normalize()).sort_index()
        c = c[~c.index.duplicated(keep="last")]
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(days)).dt.tz_localize(None).dt.normalize())
        c = c.reindex(idx).ffill()
        pnl = (c / c.iloc[0]).diff().fillna(0.0) * NOTIONAL
        pnl.iloc[0] -= fee * NOTIONAL
        pnl.iloc[-1] -= fee * NOTIONAL * float(c.iloc[-1] / c.iloc[0])
        tot = pnl if tot is None else tot + pnl
    return pd.Series(tot.to_numpy(), index=days)


def signal_rows(ds, morning, afternoon):
    """แถวเช้า (P ≥ 0.5) และบ่าย (P > 0.5) + ผลตอบแทนต้นรอบ→ปลายรอบ"""
    m = morning.assign(up=morning["p"] >= 0.5)
    a = afternoon.assign(up=afternoon["p"] > 0.5)
    sig = pd.concat([m, a], ignore_index=True)[KEYS + ["up"]]
    x = sig.merge(ds[KEYS + ["p_start", "p_end", "direction"]], on=KEYS)
    x = x[x["p_start"].notna() & x["p_end"].notna()].copy()
    x["ret"] = x["p_end"] / x["p_start"] - 1
    return x.sort_values(KEYS).reset_index(drop=True)


def main():
    s17, s18 = _module("17_label_variants.py"), _module("18_score_round4.py")
    rules = load_rules()
    tz = rules["timezone"]
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    RES.mkdir(parents=True, exist_ok=True)

    aud = s18.load("C1235", "E12_fd_lr", tz)
    e15 = pd.read_csv(OUTPUT_DIR / "experiments" / "E15" / "predictions.csv")
    e15["date"] = pd.to_datetime(e15["date"], utc=True).dt.tz_convert(tz)
    mlp = e15[e15["model"] == "mlp"]
    wf = signal_rows(ds, aud[aud["session"] == "morning"][KEYS + ["p_up"]].rename(columns={"p_up": "p"}),
                     mlp[(mlp["split"] == "wf") & (mlp["session"] == "afternoon")][KEYS + ["p"]])

    fd = next(v for v in load_experiments()["E12"]["variants"] if v["name"] == "fd_lr")
    fd = {**copy.deepcopy(fd), "params": {**fd.get("params", {}), "permutation": False}, "sessions": ["morning"]}
    ho_am = run_variant(ds, fd, rules, holdout_eval=True)
    ho = signal_rows(ds, ho_am[KEYS + ["p_up"]].rename(columns={"p_up": "p"}),
                     mlp[(mlp["split"] == "holdout") & (mlp["session"] == "afternoon")][KEYS + ["p"]])

    live = pd.read_csv(OUTPUT_DIR / "live_vm" / "live_scored.csv")
    live["date"] = pd.to_datetime(live["date"], utc=True).dt.tz_convert(tz)
    live = live[(live["date"] >= pd.Timestamp("2026-11-09", tz=tz)) & live["p_end"].notna()].copy()
    live["up"] = live["answer"] == 1
    live["p_start"] = live["p_end"] / np.exp(live["r"])
    live["ret"] = live["p_end"] / live["p_start"] - 1

    tables, diffs, curves = [], [], {}
    for period, rows in [("walk-forward", wf), ("holdout 14–29 ต.ค.", ho), ("ส่งจริง 9–13 พ.ย.", live)]:
        if rows.empty:
            continue
        days = np.sort(rows["date"].unique())
        for fee in (0.0, FEE):
            b2 = buy_hold(days, fee) if period != "ส่งจริง 9–13 พ.ย." else None
            for label, sel, k in [("เช้า", rows["session"] == "morning", 2), ("บ่าย", rows["session"] == "afternoon", 2), ("รวม", rows["session"].notna(), 4)]:
                t, diff, series = summarize(rows[sel.to_numpy()], period, label, fee, k, b2 if label == "รวม" else None)
                tables.append(t)
                if diff:
                    diffs.append(diff)
                if period == "walk-forward" and label == "รวม":
                    curves[f"{fee:.4%}"] = {k_: v.cumsum() for k_, v in series.items()}   # ผลตอบแทนสะสมต่อเงินลงทุน (ไม่ทบต้น)
    t = pd.concat(tables, ignore_index=True)
    t.to_csv(RES / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diffs).to_csv(RES / "sharpe_diff.csv", index=False, encoding="utf-8-sig")
    for fee, c in curves.items():
        e = pd.DataFrame({k: v.to_numpy() for k, v in c.items()}, index=pd.to_datetime(pd.Series(next(iter(c.values())).index)).dt.strftime("%Y-%m-%d"))
        e.index.name = "date"
        e.to_csv(RES / f"equity_wf_fee{'0' if fee == '0.0000%' else 'on'}.csv", encoding="utf-8-sig")
    counts = pd.DataFrame([{"ช่วง": p, "รอบ": s, "แถว": int((r["session"] == s).sum()), "ทายขึ้น": int(r.loc[r["session"] == s, "up"].sum())}
                           for p, r in [("walk-forward", wf), ("holdout", ho), ("ส่งจริง", live)] for s in ("morning", "afternoon")])
    counts.to_csv(RES / "signal_counts.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(counts.to_string(index=False))
    print(t.round(4).to_string(index=False))
    print(pd.DataFrame(diffs).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
