"""E28: pretrain แล้ว fine-tune แยกเช้า/บ่ายด้วยข้อมูลที่ไม่ซ้อนกัน — เกณฑ์ใน results/e28_preregistration.md · รายงานเท่านั้น

    python experiments/56_e28_disjoint_finetune.py --one        # วัดเวลา 1 บล็อก (รันได้ทุกเมื่อ)
    python experiments/56_e28_disjoint_finetune.py --part run   # รันเต็ม: หลังงานเย็นวันศุกร์ 13 พ.ย. 2026 เท่านั้น
    python experiments/56_e28_disjoint_finetune.py --part report
→ outputs/experiments/E28/ · results/e28/ · ไม่แตะระบบ live
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import time
import warnings

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forecast.config import OUTPUT_DIR, ROOT, load_experiments, load_rules
from forecast.evaluation.splits import holdout_split, walk_forward_splits
from forecast.models.finetune_lr import (KAPPAS, MIN_DAYS, PretrainFineTune, Scaler, fit_finetune, fit_pretrain, inner_blocks,
                                         logloss)

warnings.filterwarnings("ignore")
KEYS = ["date", "symbol", "session"]
SESSIONS = (("morning", "เช้า"), ("afternoon", "บ่าย"))
FRAC, GAP = 0.70, 5
SENS = (0.60, 0.80)
CONF, CONF_MECH = 0.997, 0.95
FULL_RUN_FROM = pd.Timestamp("2026-11-13 18:00", tz="Asia/Bangkok")
OUT, RES = OUTPUT_DIR / "experiments" / "E28", ROOT / "results" / "e28"


def _module(name):
    spec = importlib.util.spec_from_file_location("m" + name[:2], ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s31 = _module("31_e14_finetune.py")
PM_ONLY = s31.PM_ONLY


class DisjointFineTune(PretrainFineTune):
    """เหมือน PretrainFineTune แต่ scaler ของ feature เฉพาะบ่ายในแต่ละ fold ของ inner CV fit จากแถว ft ก่อน cut เท่านั้น
    · เก็บ β₀ / intercept ของ pretrain ไว้ตรวจ"""

    def fit(self, pre, ft, pre_cols, ft_cols, const, pm_only=None, inner_val=None):
        self.fold_extra = []
        super().fit(pre, ft, pre_cols, ft_cols, const, pm_only=pm_only, inner_val=inner_val)
        Xp = pre["X"][pre_cols].to_numpy(float)
        self.beta_pre, self.a_pre = fit_pretrain(self.scaler(Xp), pre["y"], self.C)
        self.start = self._start(self.beta_pre, self.a_pre, pre_cols, ft_cols, const)
        return self

    def _choose_kappa(self, pre, ft, inner_val):
        val = inner_val or ft
        scores = {k: [] for k in KAPPAS}
        Xpre = pre["X"][self.pre_cols].to_numpy(float)
        for tr, va in inner_blocks(val["dates"]):
            cut = np.min(va)
            a = pre["dates"] < cut
            if len(np.unique(pre["y"][a])) < 2:
                continue
            f = ft["dates"] < cut
            if not f.any():
                continue
            s = Scaler(Xpre[a])
            beta_pre, a_pre = fit_pretrain(s(Xpre[a]), pre["y"][a], self.C)
            keep_s, keep_e = self.scaler, self.extra_scaler
            self.scaler = s
            if self.extra:
                self.extra_scaler = Scaler(ft["X"][self.extra].to_numpy(float)[f])
                self.fold_extra.append({"cut": cut, "mean": self.extra_scaler.mean.copy(), "std": self.extra_scaler.std.copy()})
            beta0, a0 = self._start(beta_pre, a_pre, self.pre_cols, self.ft_cols, self.const)
            Zf = self._z(ft["X"].iloc[np.flatnonzero(f)])
            m = np.isin(val["dates"], va)
            Zv = self._z(val["X"].iloc[np.flatnonzero(m)])
            self.scaler, self.extra_scaler = keep_s, keep_e
            days_ok = len(np.unique(ft["dates"][f])) >= self.ft_days_min
            for k in KAPPAS:
                b, i = fit_finetune(Zf, ft["y"][f], beta0, a0, k / (2 * self.C)) if (days_ok and np.isfinite(k)) else (beta0, a0)
                scores[k].append(logloss(val["y"][m], 1 / (1 + np.exp(-(Zv @ b + i)))).mean())
        mean = {k: np.mean(v) for k, v in scores.items() if v}
        if not mean:
            return np.inf, np.nan
        best = min(mean, key=mean.get)
        return best, mean[best]


def split_days(labeled_days, start, frac=FRAC, gap=GAP):
    """วันที่มีป้ายก่อน start (รวมสองรอบ) → (วัน pretrain, วัน fine-tune) · เว้น gap วันทำการระหว่างกัน"""
    ud = np.unique(labeled_days[labeled_days < start])
    cut = int(np.floor(frac * len(ud)))
    return ud[:max(cut - gap, 0)], ud[cut:]


def session_cols(shared, session):
    return (shared + ["d_sess_pm", "d_sym_advanc"], shared + ["d_sym_advanc"] + (PM_ONLY if session == "afternoon" else []),
            {"d_sess_pm": 1.0 if session == "afternoon" else 0.0})


def block(d, shared, session, test_dates, kind="D-prime", frac=FRAC):
    """ฝึกจากวันก่อนบล็อกทดสอบเท่านั้น · pretrain และ fine-tune ไม่ซ้อนกัน → แถวผล + ข้อมูลบล็อก"""
    start = np.min(test_dates)
    tr = d[(d["date"].to_numpy() < start) & d["direction"].notna()]
    pre_days, ft_days = split_days(tr["date"].to_numpy(), start, frac)
    pre = tr[tr["date"].isin(pre_days)]
    ft = tr[(tr["session"] == session) & tr["date"].isin(ft_days)]
    te = d[(d["session"] == session) & d["date"].isin(test_dates)]
    pre_cols, ft_cols, const = session_cols(shared, session)
    out = te[KEYS + ["direction"]].copy()
    meta = {"session": session, "test_start": start, "n_pre_days": len(pre_days), "n_pre_am": pre.loc[pre["session"] == "morning", "date"].nunique(),
            "n_pre_pm": pre.loc[pre["session"] == "afternoon", "date"].nunique(), "n_ft_days": ft["date"].nunique(),
            "max_pre": pre["date"].max() if len(pre) else pd.NaT, "min_ft": ft["date"].min() if len(ft) else pd.NaT,
            "max_ft": ft["date"].max() if len(ft) else pd.NaT}
    if len(ft) == 0 or len(pre) == 0 or pre["direction"].eq(1).nunique() < 2:
        out["p_up"], meta["kappa"], meta["inner_ll"] = np.nan, np.nan, np.nan
    else:
        m = DisjointFineTune()
        if kind == "Pre70":
            m.ft_days_min = 10 ** 9
        m.fit(s31.part(pre), s31.part(ft), pre_cols, ft_cols, const)
        out["p_up"], meta["kappa"], meta["inner_ll"] = m.predict(te), m.kappa, m.inner_ll
    meta["common"] = bool(meta["n_ft_days"] >= MIN_DAYS and meta["n_pre_am"] >= MIN_DAYS and meta["n_pre_pm"] >= MIN_DAYS)
    out["common"] = meta["common"]
    return out, meta


def jobs_for(d, rules, holdout=False, holdout_days=None):
    ev = rules["evaluation"]
    jobs = []
    for session, _ in SESSIONS:
        dates = d.loc[d["session"] == session, "date"].to_numpy()
        splits = (holdout_split(dates, holdout_days or ev["holdout_days"]) if holdout else
                  walk_forward_splits(dates, ev["test_block_days"], ev["min_train_days"], ev["holdout_days"]))
        jobs += [(session, te) for _, te in splits]
    return jobs


def run(d, shared, rules, kind="D-prime", frac=FRAC, jobs=None):
    jobs = jobs if jobs is not None else jobs_for(d, rules)
    res = Parallel(n_jobs=-1)(delayed(block)(d, shared, s, te, kind, frac) for s, te in jobs)
    pred = pd.concat([r[0] for r in res], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return pred, pd.DataFrame([r[1] for r in res])


def check_splits(meta):
    """ทุกบล็อก: max(pre) < min(ft) ≤ max(ft) < test"""
    m = meta.dropna(subset=["max_pre", "min_ft"])
    assert (m["max_pre"] < m["min_ft"]).all() and (m["max_ft"] < m["test_start"]).all(), "ช่วง pretrain / fine-tune / test ซ้อนกัน"


def prepare(rules):
    s17 = _module("17_label_variants.py")
    from forecast.evaluation.runner import load_or_build_dataset
    end = pd.Timestamp(s17.WF_LAST_DAY).tz_localize(rules["timezone"]) + pd.Timedelta(hours=23, minutes=59)
    ds = load_or_build_dataset(rules, as_of=end, tag="_wf")
    d, shared = s31.aligned(ds, load_experiments())
    return ds, d, shared


def full_run_allowed(now=None):
    return (now or pd.Timestamp.now(tz="Asia/Bangkok")) >= FULL_RUN_FROM


def one(rules):
    ds, d, shared = prepare(rules)
    jobs = jobs_for(d, rules)
    t0 = time.perf_counter()
    _, meta = run(d, shared, rules, "D-prime", FRAC, jobs[-1:])
    sec = time.perf_counter() - t0
    m = meta.iloc[0]
    # โครงสร้างการแบ่งทุกบล็อก (ไม่ฝึก ไม่ดูผล)
    rows = []
    for s, te in jobs:
        start = np.min(te)
        tr = d[(d["date"].to_numpy() < start) & d["direction"].notna()]
        pre_days, ft_days = split_days(tr["date"].to_numpy(), start)
        pre = tr[tr["date"].isin(pre_days)]
        n_ft = tr[(tr["session"] == s) & tr["date"].isin(ft_days)]["date"].nunique()
        rows.append({"session": s, "common": n_ft >= MIN_DAYS and pre.loc[pre["session"] == "morning", "date"].nunique() >= MIN_DAYS
                     and pre.loc[pre["session"] == "afternoon", "date"].nunique() >= MIN_DAYS})
    c = pd.DataFrame(rows).groupby("session")["common"].agg(["sum", "size"])
    print(f"1 บล็อก ({m['session']}, เริ่ม {str(m['test_start'])[:10]}): {sec:.1f} วิ · κ {m['kappa']} · pre {m['n_pre_days']} วัน "
          f"(เช้า {m['n_pre_am']} บ่าย {m['n_pre_pm']}) · ft {m['n_ft_days']} วัน")
    print("บล็อกใน common window / ทั้งหมด:", c.to_dict("index"))
    print(f"บล็อกทั้งหมด {len(jobs)} · ประมาณเวลารันเต็ม (3 สัดส่วน + Pre70 · {os.cpu_count()} แกน): "
          f"{sec * len(jobs) * 4 / max(1, (os.cpu_count() or 1)) / 60:.1f}–{sec * len(jobs) * 4 / 8 / 60:.1f} นาที")


def full(rules):
    if not full_run_allowed():
        raise SystemExit(f"ยังรันเต็มไม่ได้: รอหลัง {FULL_RUN_FROM:%Y-%m-%d %H:%M} (results/e28_preregistration.md)")
    ds, d, shared = prepare(rules)
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, frac, name in [("D-prime", FRAC, "dprime"), ("Pre70", FRAC, "pre70")] + [("D-prime", f, f"dprime_{int(f * 100)}") for f in SENS]:
        t0 = time.perf_counter()
        pred, meta = run(d, shared, rules, kind, frac)
        check_splits(meta)
        pred.to_csv(OUT / f"{name}_predictions.csv", index=False)
        meta.to_csv(OUT / f"{name}_blocks.csv", index=False)
        print(name, len(pred), f"{time.perf_counter() - t0:.0f} วิ", flush=True)
    for kind, name in [("D-prime", "dprime"), ("Pre70", "pre70")]:
        h, hm = run(d, shared, rules, kind, FRAC, jobs_for(d, rules, holdout=True))
        h.to_csv(OUT / f"{name}_holdout.csv", index=False)
    s48 = _module("48_e23_skip_fx.py")
    new_ds = s48._load_new(rules)
    dn, shared_n = s31.aligned(new_ds, load_experiments())
    days_new = int((np.sort(new_ds["date"].unique()) > np.datetime64(pd.Timestamp("2026-10-29", tz=rules["timezone"]))).sum())
    for kind, name in [("D-prime", "dprime"), ("Pre70", "pre70")]:
        n, _ = run(dn, shared_n, rules, kind, FRAC, jobs_for(dn, rules, holdout=True, holdout_days=days_new))
        n.to_csv(OUT / f"{name}_new.csv", index=False)
    print("รันเต็มเสร็จ · ใช้ --part report ต่อ", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--part", nargs="+", default=[])
    args = ap.parse_args()
    rules = load_rules()
    if args.one:
        one(rules)
    if "run" in args.part:
        full(rules)
    if "report" in args.part:
        if not full_run_allowed():
            raise SystemExit("ยังไม่มีผลรันเต็ม")
        raise SystemExit("ส่วนรายงานจะเขียนหลังรันเต็ม ตามลำดับงานข้อ 8 ของ prompt")


if __name__ == "__main__":
    main()
