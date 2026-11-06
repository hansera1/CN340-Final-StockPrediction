"""E18: ดึงข้อมูลต่างประเทศจาก Yahoo Finance และตรวจข้อมูล V1, V2, V5, V7, V9

    python experiments/36_e18_fetch.py
→ data/raw/external/<ticker>_1d.csv, THB_60m.csv · results/e18_data_validation.md
"""
import io
import urllib.request

import _setup  # noqa: F401
import numpy as np
import pandas as pd
import yfinance as yf

from forecast.config import RAW_DIR, ROOT

OUT = RAW_DIR / "external"
TICKERS = ["^GSPC", "THD", "^N225", "^KS11", "THB=X", "^SET50"]
START = "2023-06-01"


def safe(t):
    return t.replace("^", "").replace("=", "_")


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    for t in TICKERS:
        df = yf.download(t, start=START, interval="1d", auto_adjust=False, progress=False, multi_level_index=False)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(OUT / f"{safe(t)}_1d.csv")
        tables[t] = df
        print(t, len(df), df.index.min().date(), df.index.max().date(), flush=True)
    h = yf.download("THB=X", period="730d", interval="60m", auto_adjust=False, progress=False, multi_level_index=False)
    h.to_csv(OUT / "THB_60m.csv")
    print("THB 60m", len(h), h.index.min(), h.index.max(), flush=True)
    return tables


def fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=40).read().decode()
    d = pd.read_csv(io.StringIO(raw))
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"])
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna().set_index("date")["v"]


def compare_close(tables, ticker, series):
    try:
        f = fred(series)
    except Exception as e:
        return f"**ยังไม่ได้ตรวจ** เข้า FRED ไม่ได้ ({type(e).__name__})"
    j = pd.concat([tables[ticker]["Close"], f], axis=1, keys=["y", "f"]).dropna()
    j = j[j.index >= "2023-10-01"]
    share = ((j["y"] / j["f"] - 1).abs() <= 1e-4).mean()
    return f"{len(j)} วัน · ต่าง ≤ 0.01% = {share:.2%} → **{'ผ่าน' if share >= .99 else 'ไม่ผ่าน'}**"


def main():
    tables = fetch()
    lines = ["# E18 — ผลตรวจข้อมูล", "", f"ดึงจาก Yahoo Finance · ตั้งแต่ {START}", "", "| ข้อ | ผล |", "|---|---|"]
    lines.append(f"| V1 S&P 500 กับ FRED SP500 | {compare_close(tables, '^GSPC', 'SP500')} |")
    lines.append(f"| V2 Nikkei กับ FRED NIKKEI225 | {compare_close(tables, '^N225', 'NIKKEI225')} |")
    lines.append("| V3 ราคาเปิด Nikkei กับเว็บ Nikkei | **ยังไม่ได้ตรวจ** ต้องเปิดดูด้วยตา |")
    lines.append("| V4 ราคาเปิด KOSPI กับเว็บ KRX | **ยังไม่ได้ตรวจ** ต้องเปิดดูด้วยตา |")
    for t in ["^N225", "^KS11"]:
        d = tables[t]
        same = (d["Open"] == d["Close"].shift(1)).mean()
        lines.append(f"| V5 {t} Open = Close เมื่อวาน | {same:.2%} ของ {len(d)} วัน → **{'ผ่าน' if same <= .05 else 'ไม่ผ่าน'}** |")
    lines.append("| V6 THD กับเว็บ iShares | **ยังไม่ได้ตรวจ** ต้องเปิดดูด้วยตา |")
    try:
        f = fred("DEXTHUS")
        j = pd.concat([np.log(tables["THB=X"]["Close"]).diff(), np.log(f).diff()], axis=1, keys=["y", "f"]).dropna()
        j = j[j["f"].abs() > 1e-3]
        same = (np.sign(j["y"]) == np.sign(j["f"])).mean()
        lines.append(f"| V7 ทิศทาง THB=X กับ FRED DEXTHUS | {len(j)} วันที่ขยับเกิน 0.1% · ตรง {same:.2%} → **{'ผ่าน' if same >= .85 else 'ไม่ผ่าน'}** |")
    except Exception as e:
        lines.append(f"| V7 ทิศทาง THB=X กับ FRED DEXTHUS | **ยังไม่ได้ตรวจ** เข้า FRED ไม่ได้ ({type(e).__name__}) |")
    lines.append("| V8 การจับคู่เวลา | test อัตโนมัติ `tests/test_e18.py` |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "e18_data_validation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
