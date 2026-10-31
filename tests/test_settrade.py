import pandas as pd

from forecast.data.io import drop_unfinished, merge_into_raw
from forecast.data.settrade import candles_to_frame


def test_candles_and_merge(rules, tmp_path):
    t0 = int(pd.Timestamp("2026-10-28 10:00", tz="Asia/Bangkok").timestamp())
    res = {"time": [t0, t0 + 60, t0 + 120], "open": [1, 2, 3], "high": [1, 2, 3],
           "low": [1, 2, 3], "close": [1.0, 2.0, 3.0], "volume": [10, 20, 30]}
    df = candles_to_frame(res, rules)
    assert df["timestamp"].iat[0] == pd.Timestamp("2026-10-28 10:00")    # เวลาไทย ไม่มี tz

    # ดึง ณ 10:02:30: แบบ start แท่ง 10:02 (จบ 10:03) ยังไม่จบ; แบบ end แท่ง 10:02 จบแล้ว
    now = pd.Timestamp("2026-10-28 10:02:30")
    assert len(drop_unfinished(df, {**rules, "bar_timestamp": "start"}, now, 1)) == 2
    assert len(drop_unfinished(df, {**rules, "bar_timestamp": "end"}, now, 1)) == 3

    path = tmp_path / "X_1m.csv"
    assert merge_into_raw(path, df.iloc[:2]) == (2, 0)
    changed = df.copy()
    changed.loc[0, "close"] = 9.0
    assert merge_into_raw(path, changed) == (1, 1)                         # เพิ่มแถวใหม่ ไม่แก้ของเดิม
    assert pd.read_csv(path)["close"].tolist() == [1.0, 2.0, 3.0]
