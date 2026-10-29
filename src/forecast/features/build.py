"""Features ของหุ้นตัวเดียว ต่อ (วัน, รอบ) โดยใช้เฉพาะข้อมูลที่จบแล้วก่อนจุดตัด

- รายวัน: shift(1) เสมอ → แถววันนี้เห็นแค่ถึงเมื่อวาน
- ภายในวัน: เฉพาะแท่งที่ bar_end <= cutoff (รอบเช้าตัด 09:15 จึงยังไม่มี)
- ช่วงเวลาเดียวกันในอดีต: r ของวันก่อน ๆ (รู้หลังปิดรอบของวันนั้นแล้ว)
"""
import numpy as np
import pandas as pd

from forecast.config import session_times
from forecast.data.asof import bars_until, last_close
from forecast.features.spectral import band_powers


def daily_table(bars):
    g = bars.groupby("date", sort=True)
    d = pd.DataFrame({
        "open": g["open"].first() if "open" in bars else g["close"].first(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    d["ret"] = np.log(d["close"]).diff()
    return d


def daily_features(daily):
    """ค่า ณ สิ้นวันของแต่ละวัน — ตอนใช้ต้องดึงแถวของ "วันทำการก่อนหน้า" เท่านั้น"""
    lc = np.log(daily["close"])
    f = pd.DataFrame(index=daily.index)
    f["f_ret_1d"] = daily["ret"]
    f["f_ret_5d"] = lc - lc.shift(5)
    f["f_ret_20d"] = lc - lc.shift(20)
    f["f_rv_5d"] = daily["ret"].rolling(5).std()
    f["f_rv_20d"] = daily["ret"].rolling(20).std()
    f["f_rvol_1d"] = daily["volume"] / daily["volume"].rolling(20).mean()
    f["f_dist_ma20"] = lc - np.log(daily["close"].rolling(20).mean())
    f["prev_close"] = daily["close"]
    return f


def before(table, date):
    """แถวล่าสุดของ table ที่ index < date (วันทำการก่อนหน้า) หรือ None"""
    i = table.index.searchsorted(date, side="left") - 1
    return table.iloc[i] if i >= 0 else None


def intraday_features(day_bars, cutoff, prev_close):
    b = bars_until(day_bars, cutoff)
    if b.empty:
        return {}
    first_open = b["open"].iat[0] if "open" in b else b["close"].iat[0]
    last = b["close"].iat[-1]
    lr = np.log(b["close"]).diff().dropna()
    before30, _ = last_close(b, cutoff - pd.Timedelta(minutes=30))
    out = {
        "f_r_open_to_cut": np.log(last / first_open),
        "f_gap_open": np.log(first_open / prev_close) if prev_close else np.nan,
        "f_r_last30": np.log(last / (before30 if np.isfinite(before30) else first_open)),
        "f_rv_intra": float(np.sqrt((lr ** 2).sum())),
        "f_absr_intra": float(lr.abs().mean()) if len(lr) else np.nan,
        "cumvol_cut": float(b["volume"].sum()),
    }
    if "high" in b and "low" in b:
        out["f_hl_range_intra"] = float(np.log(b["high"].max() / b["low"].min()))
    return out


def build_features(bars, rows, rules, feature_bars=None, spectral=False):
    """rows: DataFrame[date, session, r] — r ของวันในอดีต ใช้ทำ same-window history
    แถวที่ยังไม่รู้ผล (เช่นวันนี้ในวันจริง) ให้ r = NaN ได้

    bars ใช้คำนวณ features รายวัน; feature_bars (อาจ resample แล้ว) ใช้คำนวณ features ภายในวัน
    """
    fb = bars if feature_bars is None else feature_bars
    daily = daily_table(bars)
    dfeat = daily_features(daily)
    day_groups = dict(tuple(fb.groupby("date", sort=True)))
    all_dates = sorted(day_groups)

    out = []
    for date, sess in rows[["date", "session"]].itertuples(index=False):
        t = session_times(rules, sess, date)
        rec = {"date": date, "session": sess, "cutoff_ts": t["cutoff"], "f_dow": date.dayofweek}
        prev = before(dfeat, date)
        pc = np.nan
        if prev is not None:
            rec.update(prev.drop("prev_close").to_dict())
            pc = prev["prev_close"]
        day = day_groups.get(date)
        if day is not None:
            rec.update(intraday_features(day, t["cutoff"], pc))
        if spectral:
            today = bars_until(day, t["cutoff"]) if day is not None else None
            if today is None or len(today) < 32:   # รอบเช้า: ใช้ทั้งวันของวันก่อน
                prev = [d for d in all_dates if d < date]
                today = day_groups[prev[-1]] if prev else None
            if today is not None:
                rec.update(band_powers(today["close"]))
        out.append(rec)
    f = pd.DataFrame(out)

    # ปริมาณสะสมถึงจุดตัด เทียบเวลาเดียวกันของ 20 วันก่อน (ใช้อดีตเท่านั้น)
    if "cumvol_cut" in f:
        f = f.sort_values("date")
        f["f_vol_ratio_same_time"] = f.groupby("session")["cumvol_cut"].transform(
            lambda s: s / s.shift(1).rolling(20, min_periods=5).mean())
        f = f.drop(columns="cumvol_cut")

    # ช่วงเวลาเดียวกันในอดีต + reversal ของรอบเมื่อวาน
    r = rows.pivot(index="date", columns="session", values="r").sort_index()
    r_prev = r.shift(1)
    hist = []
    for sess in r.columns:
        h = pd.DataFrame({"date": r.index, "session": sess})
        for n in (5, 10, 20):
            h[f"f_r_same_mean{n}"] = r_prev[sess].rolling(n, min_periods=max(2, n // 2)).mean().to_numpy()
        for s2 in r.columns:
            h[f"f_r_prev_{s2}"] = r_prev[s2].to_numpy()
        hist.append(h)
    f = f.merge(pd.concat(hist), on=["date", "session"], how="left")
    return f.sort_values(["date", "session"]).reset_index(drop=True)
