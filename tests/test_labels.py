import numpy as np
import pandas as pd

from forecast.data.io import load_bars
from forecast.labels.make_labels import make_labels


def test_label_prices_at_target_times(rules, raw_dir):
    bars = load_bars("KBANK", rules, raw_dir)
    lab = make_labels(bars, rules).set_index(["date", "session"])
    day = bars[bars["date"] == bars["date"].iloc[-1]].set_index("bar_end")
    date = day["date"].iloc[0]
    at = lambda hm: date + pd.Timedelta(hm)

    m = lab.loc[(date, "morning")]
    assert m["p_start"] == day.loc[at("10:01:00"), "open"]      # ราคาเปิดของแท่งแรก
    assert m["p_end"] == day.loc[at("12:30:00"), "close"]
    a = lab.loc[(date, "afternoon")]
    assert a["p_start"] == day.loc[at("14:01:00"), "open"]
    assert a["p_end"] == day.loc[at("16:30:00"), "close"]
    assert a["p_ref"] == day.loc[at("12:15:00"), "close"]       # ราคาล่าสุดที่รู้ ณ c_บ่าย
    assert np.isclose(a["r"], np.log(a["p_end"] / a["p_start"]))
    assert np.isclose(a["z"], np.log(a["p_end"] / a["p_ref"]))


def test_morning_reference_is_previous_close(rules, raw_dir):
    bars = load_bars("KBANK", rules, raw_dir)
    lab = make_labels(bars, rules)
    dates = sorted(bars["date"].unique())
    prev_close = bars[bars["date"] == dates[-2]]["close"].iloc[-1]
    row = lab[(lab["date"] == dates[-1]) & (lab["session"] == "morning")].iloc[0]
    assert row["p_ref"] == prev_close


def _start_stamped_day_with_auction(tmp_path):
    """แท่งแบบเวลาเริ่ม (10:00 … 16:29) + แท่งประมูลปิด 16:30–16:39 ราคาบนช่องราคา 0.5 บาท"""
    d = pd.Timestamp("2026-10-28")
    mins = list(pd.timedelta_range("10:00:00", "12:29:00", freq="1min")) + \
        list(pd.timedelta_range("14:00:00", "16:39:00", freq="1min"))
    rows = []
    for m in mins:
        p = 152.0 if m >= pd.Timedelta("16:30:00") else 150.0   # ปลายช่วง = ต้นช่วง → flat
        rows.append((d + m, p, p, 100))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "close", "volume"])
    df.to_csv(tmp_path / "KBANK_1min.csv", index=False)
    return tmp_path


def test_start_timestamp_auction_and_flat(rules, tmp_path):
    from forecast.data.quality import timestamp_profile
    rules = {**rules, "bar_timestamp": "start",
             "label": {**rules["label"], "price_1630": "last_trade_at_or_before",
                       "store_closing_auction": True}}
    bars = load_bars("KBANK", rules, _start_stamped_day_with_auction(tmp_path))
    prof = timestamp_profile(bars, rules)
    assert prof["bar_timestamp_guess"] == "start" and "WARNING" not in prof
    assert prof["days_with_bars_after_1630"] == "1 / 1"

    a = make_labels(bars, rules).set_index("session").loc["afternoon"]
    assert a["p_end"] == 150.0                 # ราคา ณ/ก่อน 16:30 ไม่ใช่ open ของแท่งประมูล
    assert a["p_close_auction"] == 152.0
    assert a["direction"] == 0 and a["up"] == 0   # flat = ต่าง 0 ช่องราคา

    rules["label"]["price_1630"] = "closing_auction"
    a = make_labels(bars, rules).set_index("session").loc["afternoon"]
    assert a["p_end"] == 152.0 and a["direction"] == 1
