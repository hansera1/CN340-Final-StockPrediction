"""ประกอบ dataset ของทุกหุ้น — ใช้ฟังก์ชันเดียวกันทั้ง backtest และวันจริง"""
import pandas as pd

from forecast.config import session_times
from forecast.data.io import aggregate_bars, load_bars
from forecast.features.build import build_features
from forecast.features.external import asof_join, load_external
from forecast.data.asof import last_close
from forecast.labels.make_labels import apply_official_close, daily_closes, labels_for

KEYS = ["date", "symbol", "session"]


def feature_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


def add_sigma(ds, n=20):
    """sig20 = ค่าเฉลี่ย |z| ของ (หุ้น, รอบ) เดียวกันใน n วันก่อนหน้า — σ̂ ของช่วงราคา E9 (ไม่ใช่ feature)
    z ของวันก่อน ๆ รู้ผลแล้ว ณ จุดตัดของวันนี้ (รอบบ่ายเมื่อวานจบ 16:30)"""
    ds = ds.sort_values(KEYS).copy()
    ds["sig20"] = ds.groupby(["symbol", "session"])["z"].transform(
        lambda s: s.abs().shift(1).rolling(n, min_periods=n // 2).mean())
    return ds


def session_ranges(bars, rules):
    """high / low ของแต่ละรอบ (แท่งที่เริ่มใน [เวลาเริ่ม, เวลาปลาย) ของรอบ) + เวลาที่รอบนั้นจบ"""
    rows = []
    for date, day in bars.groupby("date", sort=True):
        for sess in rules["sessions"]:
            t = session_times(rules, sess, date)
            b = day[(day["bar_start"] >= t["start"]) & (day["bar_start"] < t["end"])]
            if len(b):
                rows.append({"end_ts": t["end"], "hi": b["high"].max() if "high" in b else b["close"].max(),
                             "lo": b["low"].min() if "low" in b else b["close"].min()})
    return pd.DataFrame(rows).sort_values("end_ts")


def add_flat_features(ds, bars_by_symbol, rules, n=20):
    """E9-FLAT (Model A) — ชื่อขึ้นต้น fl_ (ไม่ปนกับชุด f_ ของ E1) ใช้เฉพาะข้อมูลที่จบก่อนจุดตัด:
    fl_exp_ticks         |r| เฉลี่ยของ (หุ้น, รอบ) เดียวกัน n วันก่อน × P(c) ÷ ช่องราคาของ P(c)
    fl_prev_range_ticks  ช่วงสูง–ต่ำของรอบล่าสุดที่จบก่อนจุดตัด (เช้า/บ่าย = บ่ายของวันทำการก่อน) ÷ ช่องราคา
    fl_flat20_dir/price  สัดส่วนวันนิ่ง (ปลาย = ต้นช่วง / ปลาย = P(c)) ของรอบเดียวกัน n วันก่อน
    fl_rv_intra          realized vol ภาคเช้าถึงจุดตัด (รอบเช้า = 0 — ใช้คู่กับตัวแปรบอกรอบ)
    fl_dow_0..4          วันในสัปดาห์แบบ one-hot
    ไม่ใช้ volume ภายในวัน (yfinance ช่วงเช้าเสีย)"""
    from forecast.evaluation.metrics import tick_size
    ds = ds.sort_values(KEYS).copy()
    tick = tick_size(ds["p_ref"])
    g = ds.groupby(["symbol", "session"])
    past = lambda s: s.shift(1).rolling(n, min_periods=n // 2).mean()
    ds["fl_exp_ticks"] = g["r"].transform(lambda s: past(s.abs())) * ds["p_ref"] / tick
    ds["fl_flat20_dir"] = g["direction"].transform(lambda s: past((s == 0).astype(float).where(s.notna())))
    same = (ds["p_end"] - ds["p_ref"]).abs().lt(1e-6).astype(float).where(ds["p_end"].notna())
    ds["fl_flat20_price"] = same.groupby([ds["symbol"], ds["session"]]).transform(past)
    ds["fl_rv_intra"] = ds["f_rv_intra"].where(ds["session"] == "afternoon", 0.0) if "f_rv_intra" in ds else 0.0
    for k in range(5):
        ds[f"fl_dow_{k}"] = (ds["date"].dt.dayofweek == k).astype(float)
    parts = []
    for sym, d in ds.groupby("symbol", sort=False):
        rng = session_ranges(bars_by_symbol[sym], rules)
        d = d.sort_values("cutoff_ts")
        m = pd.merge_asof(d[["cutoff_ts"]].reset_index(), rng, left_on="cutoff_ts", right_on="end_ts",
                          direction="backward", allow_exact_matches=True).set_index("index")
        parts.append((m["hi"] - m["lo"]).reindex(d.index))
    ds["fl_prev_range_ticks"] = pd.concat(parts).reindex(ds.index) / tick
    return ds


def select_features(g, variant):
    """features ของ variant: ระบุรายชื่อต่อรอบใน experiments.yaml (features: {morning: [...], ...})
    หรือใช้ทุกคอลัมน์ f_ ถ้าไม่ระบุ"""
    spec = variant.get("features")
    if not spec:
        return feature_cols(g)
    names = spec[g["session"].iat[0]] if isinstance(spec, dict) else spec
    return [c for c in names if c in g]


def excluded_cols(df, rules):
    """rules.yaml → features.exclude (ชื่อไม่มี f_) — ตัดทั้ง f_<name> และ f_x_<SYM>_<name>"""
    names = rules.get("features", {}).get("exclude", [])
    return [c for c in feature_cols(df) for n in names
            if c == f"f_{n}" or (c.startswith("f_x_") and c.endswith(f"_{n}"))]


def assemble_dataset(rules, raw_dir=None, ext_dir=None, feature_minutes=None, spectral=False,
                     as_of=None, live_row=None, interval=None, bars_by_symbol=None, daily_dir=None):
    """interval: ไฟล์ดิบที่ใช้ (default = rules.yaml → data.interval)
    feature_minutes: รวมแท่งเป็นกี่นาทีก่อนคำนวณ features ภายในวัน (None = ใช้แท่งตามไฟล์)
    as_of: ตัดข้อมูลดิบทุกอย่างหลังเวลานี้ทิ้งก่อนเริ่ม (ใช้ในวันจริง)
    live_row: (date, session) ที่ยังไม่รู้ผล ให้เพิ่มเป็นแถวที่ไม่มี label
    bars_by_symbol: แท่งที่เตรียมไว้แล้ว (วันจริง: ประวัติ 60m + แท่ง 1m วันนี้ที่รวมแล้ว) แทนการอ่านไฟล์
    daily_dir: โฟลเดอร์ที่มี settrade/ และ yfinance/ ของแท่งรายวัน (None = data/raw; ข้อมูลจำลอง = ไม่มี)
    """
    if daily_dir is None and raw_dir is not None:
        daily_dir = raw_dir / "_none"
    per_symbol, used_bars = {}, {}
    for sym in rules["symbols"]:
        bars = bars_by_symbol[sym] if bars_by_symbol else load_bars(sym, rules, raw_dir, interval)
        if as_of is not None:
            bars = bars[bars["bar_end"] <= as_of].reset_index(drop=True)
        used_bars[sym] = bars
        labels = labels_for(bars, rules, interval or rules["data"]["interval"], daily_dir)
        if live_row is not None:
            # แถวที่ยังไม่รู้ผลต้องมี P(c) นิยามเดียวกับ backtest (fl_* ของโมเดลราคาใช้ค่านี้)
            date, sess = live_row
            labels = labels[labels["date"] < date]
            t = session_times(rules, sess, date)
            p_ref, ref_time = last_close(bars, t["cutoff"])
            row = pd.DataFrame([{"date": date, "symbol": sym, "session": sess, "cutoff_ts": t["cutoff"],
                                 "p_ref": p_ref, "ref_time": ref_time}])
            row = apply_official_close(row, rules, daily_closes(sym, daily_dir), [*bars["date"].unique(), date])
            labels = pd.concat([labels, row], ignore_index=True)
        fb = aggregate_bars(bars, feature_minutes, rules) if feature_minutes else bars
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

    drop = excluded_cols(ds, rules)
    if drop:
        ds = ds.drop(columns=drop)

    ds = add_flat_features(add_sigma(ds), used_bars, rules)
    # E12 (สูตร CN340): ไฟล์รายวันอยู่ใต้ data/raw จริงเท่านั้น — ข้อมูลจำลอง (raw_dir) ไม่มี fd_
    from forecast.features.cn340 import add_cn340_features
    return add_cn340_features(ds, used_bars, rules, daily_dir).reset_index(drop=True)
