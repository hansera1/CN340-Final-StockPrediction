"""E27 ตรวจเพิ่มหลังเห็นผล: แยกวัน XD ออกจากผลรวม · รายงานเท่านั้น · python experiments/55_e27_xd_check.py → results/e27/xd_*.csv"""
import importlib.util

import _setup  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast.config import EXTERNAL_DIR, OUTPUT_DIR, ROOT, load_rules  # noqa: E402


def mod(n):
    spec = importlib.util.spec_from_file_location("m" + n[:2], ROOT / "experiments" / n)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s48, s27, s37 = mod("48_e23_skip_fx.py"), mod("54_e27_gap_analysis.py"), mod("37_e18_overnight.py")
rules = load_rules()
tz = rules["timezone"]
ds, exps, s31, _, s38, make_fx = s48.prepare(rules)
am, _ = s38.morning_frame(ds, make_fx(ds))
cal = pd.read_csv(EXTERNAL_DIR / "calendar.csv")
am["fx_xd"], am["_div"] = s27.xd_feature(am, cal)
base = list(s38.FX)
rows = []
for name, cols in [("ชุดเดิม 4 ตัว", base), ("ชุดเดิม + fx_xd", base + ["fx_xd"])]:
    gh = s27.gap_with(am, cols, s38)
    j = am.merge(gh, on=s27.KEYS)
    for sym, g in j.groupby("symbol"):
        g = g[s38.wf_rows(g, rules)]
        for lab, h in [("ทุกวัน", g), ("ไม่รวมวัน XD", g[g["_div"] == 0])]:
            r, lo, hi, n = s37.boot_corr(h["g_hat"].to_numpy(), h["gap"].to_numpy(), h["date"].to_numpy(), 2000)
            rows.append({"set": name, "symbol": sym, "days": lab, "n": n, "corr": r, "lo": lo, "hi": hi})
c = pd.DataFrame(rows)
c.to_csv(ROOT / "results" / "e27" / "xd_corr_split.csv", index=False, encoding="utf-8-sig")

x = pd.read_csv(OUTPUT_DIR / "experiments" / "E18b" / "points.csv")
x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_convert(tz)
x = x.merge(am[s27.KEYS + ["_div"]], on=s27.KEYS)
pe, pc, div = x["p_end"].to_numpy(float), x["pc"].to_numpy(float), x["_div"].to_numpy(float)
xd = div > 0
pra = x["pt_PR-A"].to_numpy(float)
ps = x["p_start"].to_numpy(float)
out = []
for lab, pt in [("PR-A", pra), ("P(c)", pc), ("PR-A หักปันผลวัน XD", pra - div), ("P(c) หักปันผลวัน XD", pc - div), ("P(10:00)", ps)]:
    e = pe - pt
    out.append({"model": lab, "rmse_all": np.sqrt((e ** 2).mean()), "mae_all": np.abs(e).mean(), "rmse_no_xd": np.sqrt((e[~xd] ** 2).mean()), "mae_no_xd": np.abs(e[~xd]).mean()})
o = pd.DataFrame(out)
o.to_csv(ROOT / "results" / "e27" / "xd_effect.csv", index=False, encoding="utf-8-sig")
share = pd.DataFrame([{"model": "PR-A", "xd_rows": int(xd.sum()), "share_of_squared_error": float(((pe - pra)[xd] ** 2).sum() / ((pe - pra) ** 2).sum())}])
pd.set_option("display.width", 180)
print(c.round(4).to_string(index=False))
print(o.round(3).to_string(index=False))
print(share.round(4).to_string(index=False))
