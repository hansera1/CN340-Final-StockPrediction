"""ประกอบ dataset ของทุกหุ้น — ใช้ฟังก์ชันเดียวกันทั้ง backtest และวันจริง"""
import pandas as pd

from forecast.config import session_times
from forecast.data.io import load_bars, resample_bars
from forecast.features.build import build_features
from forecast.features.external import asof_join, load_external
from forecast.labels.make_labels import make_labels

KEYS = ["date", "symbol", "session"]


def feature_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


def assemble_dataset(rules, raw_dir=None, ext_dir=None, feature_minutes=1, spectral=False,
                     as_of=None, live_row=None):
    """as_of: ตัดข้อมูลดิบทุกอย่างหลังเวลานี้ทิ้งก่อนเริ่ม (ใช้ในวันจริง)
    live_row: (date, session) ที่ยังไม่รู้ผล ให้เพิ่มเป็นแถวที่ไม่มี label
    """
    per_symbol = {}
    for sym in rules["symbols"]:
        bars = load_bars(sym, rules, raw_dir)
        if as_of is not None:
            bars = bars[bars["bar_end"] <= as_of].reset_index(drop=True)
        labels = make_labels(bars, rules)
        if live_row is not None:
            date, sess = live_row
            labels = labels[labels["date"] < date]
            t = session_times(rules, sess, date)
            labels = pd.concat([labels, pd.DataFrame([{
                "date": date, "symbol": sym, "session": sess, "cutoff_ts": t["cutoff"]}])],
                ignore_index=True)
        fb = resample_bars(bars, feature_minutes)
        feats = build_features(bars, labels, rules, feature_bars=fb, spectral=spectral)
        per_symbol[sym] = labels.merge(feats.drop(columns="cutoff_ts"), on=["date", "session"], how="left")

    # ข้ามหลักทรัพย์: features ของหุ้นอีกตัว ณ จุดตัดเดียวกัน
    frames = []
    for sym, df in per_symbol.items():
        for other, odf in per_symbol.items():
            if other == sym:
                continue
            cols = [c for c in feature_cols(odf) if not c.startswith("f_x_") and c != "f_dow"]
            x = odf[["date", "session"] + cols].rename(columns={c: f"f_x_{other}_{c[2:]}" for c in cols})
            df = df.merge(x, on=["date", "session"], how="left")
        frames.append(df)
    ds = pd.concat(frames, ignore_index=True)

    for name, ext in load_external(rules, ext_dir).items():
        ds = asof_join(ds, ext, name)

    return ds.sort_values(KEYS).reset_index(drop=True)
