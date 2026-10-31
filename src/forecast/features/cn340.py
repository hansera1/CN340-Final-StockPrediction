"""E12: สูตร feature จากงานรายวันเดิม (CN340) — คำนวณจากข้อมูลก่อนจุดตัดเท่านั้น

fd_*  F-daily: technical จากแท่งรายวัน ณ ราคาปิดของวันทำการก่อนวันที่ทาย (ใช้ทั้งรอบเช้าและบ่าย)
      แท่งรายวัน = Settrade 1d (ราคาปิดทางการรวม ATC + volume) ต่อด้วย yfinance 1d ก่อนวันแรกของ Settrade (warm-up)
fh_*  F-60m: technical บนแท่ง intraday ต่อเนื่องข้ามวัน ถึงแท่งล่าสุดที่จบ ณ/ก่อนจุดตัด (ใช้กับรอบบ่ายเท่านั้น)
      ฟังก์ชันเดียวกันใช้กับแท่ง 30m / 5m ใน pilot timeframe (prefix fh30_ / fh5_)
ไม่ใช้ volume ภายในวันจาก yfinance · วันในสัปดาห์ของวันที่ทายอยู่ใน fl_dow_* แล้ว
"""
import numpy as np
import pandas as pd

from forecast.config import RAW_DIR

RET_DAYS = [1, 2, 3, 5, 10]


def _wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def macd(close, fast=12, slow=26, signal=9):
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    s = m.ewm(span=signal, adjust=False).mean()
    return m / close, s / close, (m - s) / close


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def bollinger(close, n=20, k=2):
    ma, sd = close.rolling(n).mean(), close.rolling(n).std()
    lo, hi = ma - k * sd, ma + k * sd
    return (close - lo) / (hi - lo), (hi - lo) / ma


def winsorize_expanding(s, lo=0.01, hi=0.99, min_periods=60):
    """ตัดปลายด้วย quantile แบบ expanding (อดีตถึงวันนั้น) — ไม่ใช้ข้อมูลอนาคต"""
    ql = s.expanding(min_periods).quantile(lo)
    qh = s.expanding(min_periods).quantile(hi)
    return s.clip(ql, qh).where(ql.notna(), s)


def load_daily(symbol, base=None):
    """Settrade 1d ต่อด้วย yfinance 1d เฉพาะวันก่อนวันแรกของ Settrade (yfinance 1d ปี 2024 มีวันเสีย)"""
    base = base or RAW_DIR
    st = pd.read_csv(base / "settrade" / f"{symbol}_1d.csv", parse_dates=["timestamp"])
    parts = [st]
    yf_path = base / "yfinance" / f"{symbol}_1d.csv"
    if yf_path.exists():
        yf = pd.read_csv(yf_path, parse_dates=["timestamp"])
        parts.insert(0, yf[yf["timestamp"] < st["timestamp"].min()])
    d = pd.concat(parts, ignore_index=True)[["timestamp", "open", "high", "low", "close", "volume"]]
    d["timestamp"] = d["timestamp"].dt.normalize()
    return d.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def daily_features(d):
    """ค่า ณ ราคาปิดของแต่ละวัน (แถว t ใช้ข้อมูลถึงวัน t) → จะถูกจับคู่กับวันทำการถัดไป"""
    o, h, l, c, v = (d[k].astype(float) for k in ["open", "high", "low", "close", "volume"])
    f = pd.DataFrame({"timestamp": d["timestamp"]})
    for n in RET_DAYS:
        f[f"fd_ret_{n}"] = winsorize_expanding(c.pct_change(n))
    f["fd_hl_range"] = (h - l) / c
    f["fd_oc_change"] = winsorize_expanding((c - o) / o)
    f["fd_close_pos"] = ((c - l) / (h - l)).where(h > l, 0.5)
    sma5, sma20 = c.rolling(5).mean(), c.rolling(20).mean()
    ema5, ema20 = c.ewm(span=5, adjust=False).mean(), c.ewm(span=20, adjust=False).mean()
    f["fd_c_sma5"], f["fd_c_sma20"] = c / sma5, c / sma20
    f["fd_c_ema5"], f["fd_c_ema20"] = c / ema5, c / ema20
    f["fd_sma5_sma20"] = sma5 / sma20
    f["fd_rsi14"] = rsi(c)
    f["fd_macd"], f["fd_macd_signal"], f["fd_macd_hist"] = macd(c)
    f["fd_bb_pos"], f["fd_bb_width"] = bollinger(c)
    f["fd_atr14"] = atr(h, l, c) / c
    up = (c.diff() > 0).astype(int)
    f["fd_up_streak"] = up.groupby((up == 0).cumsum()).cumsum()
    f["fd_c_max20"] = c / c.rolling(20).max() - 1
    f["fd_c_min20"] = c / c.rolling(20).min() - 1
    f["fd_vol_ratio"] = (v / v.rolling(20).mean()).where(v > 0)
    return f


def bar_features(bars, prefix="fh"):
    """technical บนแท่ง intraday ต่อเนื่องข้ามวัน — ค่า ณ bar_end ของแต่ละแท่ง"""
    b = bars.sort_values("bar_end").reset_index(drop=True)
    c, h, l = b["close"].astype(float), b["high"].astype(float), b["low"].astype(float)
    f = pd.DataFrame({"bar_end": b["bar_end"]})
    f[f"{prefix}_rsi14"] = rsi(c)
    f[f"{prefix}_macd_hist"] = macd(c)[2]
    f[f"{prefix}_bb_pos"] = bollinger(c)[0]
    f[f"{prefix}_c_sma20"] = c / c.rolling(20).mean()
    f[f"{prefix}_atr14"] = atr(h, l, c) / c
    return f


def asof_bar_features(ds, bars_by_symbol, prefix="fh"):
    """ค่าของแท่งล่าสุดที่จบ ณ/ก่อน cutoff_ts ของแต่ละแถว"""
    parts = []
    for sym, d in ds.groupby("symbol", sort=False):
        f = bar_features(bars_by_symbol[sym], prefix)
        d = d.sort_values("cutoff_ts")
        m = pd.merge_asof(d[["cutoff_ts"]].reset_index(), f, left_on="cutoff_ts", right_on="bar_end",
                          direction="backward", allow_exact_matches=True).set_index("index")
        parts.append(m.drop(columns=["cutoff_ts", "bar_end"]))
    return pd.concat(parts).reindex(ds.index)


def add_cn340_features(ds, bars_by_symbol, rules, daily_dir=None):
    """daily_dir: โฟลเดอร์ที่มี settrade/ และ yfinance/ (None = data/raw) — ไม่มีไฟล์รายวัน → ไม่มีคอลัมน์ fd_"""
    ds = ds.copy()
    parts = []
    for sym, d in ds.groupby("symbol", sort=False):
        try:
            f = daily_features(load_daily(sym, daily_dir))
        except FileNotFoundError:
            continue
        key = pd.DataFrame({"i": d.index, "day": d["date"].dt.tz_localize(None).dt.normalize()}).sort_values("day")
        # วันทำการก่อนวันที่ทาย: แท่งรายวันล่าสุดที่วันที่ < วันที่ทาย (ราคาปิดเมื่อวาน)
        m = pd.merge_asof(key, f, left_on="day", right_on="timestamp", direction="backward",
                          allow_exact_matches=False).set_index("i")
        parts.append(m.drop(columns=["day", "timestamp"]))
    if parts:
        ds = ds.join(pd.concat(parts))
    return ds.join(asof_bar_features(ds, bars_by_symbol, "fh"))
