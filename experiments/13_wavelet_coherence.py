"""ข้อ 5 (รายงานเท่านั้น ไม่ใช้เลือกโมเดล): ความสัมพันธ์ในย่านความถี่สูง

A1 wavelet coherence (Grinsted, Moore & Jevrejeva 2004) ของผลตอบแทน 1m (ทุกวันที่มี) และ 5m (60 วัน)
   คู่ KBANK–ADVANC, KBANK–SET50, ADVANC–SET50 · แยกภาคเช้า (10:00–12:30) / บ่าย (14:00–16:30)
   ไม่ต่อข้ามช่วงพักเที่ยงหรือข้ามวัน: แต่ละช่วงทำ CWT แยกกัน
   - แผนที่ coherence รายช่วง + ลูกศรเฟส + cone of influence + เส้นนัยสำคัญ 95% เทียบ red noise AR(1) (Monte Carlo)
   - สรุป: coherence เฉลี่ยตามคาบ (เฉลี่ย spectrum ที่ smooth แล้วข้ามทุกช่วง เฉพาะในกรวย COI)
     เทียบเกณฑ์ 95% จาก AR(1) surrogate ที่ผ่านขั้นตอนเดียวกัน, เฟสเฉลี่ย → ใครนำกี่นาที
     และตรวจซ้ำด้วย lagged correlation ของผลตอบแทน
A2 วันเดียวกันที่มีทั้ง 1m และ 5m: scalogram 1m / 5m วางเทียบ พร้อมแท่ง 60m เพื่อแสดงผลของ Nyquist

    python experiments/13_wavelet_coherence.py
→ results/a1/, results/a2/
"""
import argparse

import _setup  # noqa: F401
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pycwt  # noqa: E402

from forecast.config import RAW_DIR, ROOT  # noqa: E402

plt.rcParams["font.family"] = "Tahoma"
OUT = ROOT / "results"
SESS = {"morning": ("10:00", "12:30"), "afternoon": ("14:00", "16:30")}
PAIRS = [("KBANK", "ADVANC"), ("KBANK", "^SET50"), ("ADVANC", "^SET50")]
NAME = {"^SET50": "SET50"}
MOTHER = pycwt.Morlet(6)
DJ = 1 / 8


def load(sym, iv):
    b = pd.read_csv(RAW_DIR / "yfinance" / f"{sym}_{iv}.csv", parse_dates=["timestamp"])
    return b.set_index("timestamp")["close"].sort_index()


def segments(sym_a, sym_b, iv):
    """→ list of (วัน, รอบ, r_a, r_b) ผลตอบแทน log ของแท่งในรอบ (ตารางเวลาเต็ม, นาทีไม่มีซื้อขาย = ราคาเดิม)"""
    step = int(iv[:-1])
    a, b = load(sym_a, iv), load(sym_b, iv)
    out = []
    for day in sorted(set(a.index.normalize()) & set(b.index.normalize())):
        for sess, (t0, t1) in SESS.items():
            grid = pd.date_range(day + pd.Timedelta(t0 + ":00"), day + pd.Timedelta(t1 + ":00"),
                                 freq=f"{step}min", inclusive="left")
            xa = a.reindex(a.index.union(grid)).ffill().reindex(grid)
            xb = b.reindex(b.index.union(grid)).ffill().reindex(grid)
            if xa.isna().any() or xb.isna().any() or a[grid[0]:grid[-1]].size < 0.8 * len(grid):
                continue
            ra, rb = np.diff(np.log(xa.to_numpy())), np.diff(np.log(xb.to_numpy()))
            if ra.std() == 0 or rb.std() == 0:
                continue
            out.append((day, sess, ra, rb))
    return out


def scales_for(n, dt):
    s0 = 2 * dt
    J = int(np.log2(n * dt / 3 / s0) / DJ)        # คาบยาวสุด ≈ ความยาวช่วง / 3
    return s0, J


def smoothed(x, y, dt, s0, J):
    nx, ny = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    W1, sj, freq, coi, _, _ = pycwt.cwt(nx, dt, DJ, s0, J, MOTHER)
    W2 = pycwt.cwt(ny, dt, DJ, s0, J, MOTHER)[0]
    sc = sj[:, None]
    S1 = MOTHER.smooth(np.abs(W1) ** 2 / sc, dt, DJ, sj)
    S2 = MOTHER.smooth(np.abs(W2) ** 2 / sc, dt, DJ, sj)
    S12 = MOTHER.smooth(W1 * W2.conj() / sc, dt, DJ, sj)
    period = 1 / freq
    inside = period[:, None] <= coi[None, :]          # ภายในกรวย COI
    return S1, S2, S12, W1 * W2.conj(), period, coi, inside


def local(x, y, dt, s0, J):
    """coherence เฉพาะจุด (Grinsted 2004) + เฟส + mask ในกรวย COI"""
    S1, S2, S12, W12, period, coi, inside = smoothed(x, y, dt, s0, J)
    return np.abs(S12) ** 2 / (S1 * S2), np.angle(S12), period, inside


def summarize_pair(segs, dt, s0, J, mc, rng):
    """ต่อคาบ: coherence เฉลี่ย (ทุกช่วง ทุกเวลาในกรวย COI) เทียบค่าเดียวกันจาก AR(1) surrogate อิสระ,
    สัดส่วนพื้นที่ที่ coherence เกินเกณฑ์ 95% เฉพาะจุด (ถ้าไม่มีความสัมพันธ์จะได้ราว 5%) และเฟสเฉลี่ยในพื้นที่นั้น"""
    real = [local(x, y, dt, s0, J) for x, y in segs]
    period = real[0][2]
    null_local, null_mean = [], []
    for _ in range(mc):
        rep = [local(ar1_surrogate(x, len(x), rng), ar1_surrogate(y, len(y), rng), dt, s0, J) for x, y in segs]
        vals = np.concatenate([np.where(ins, w, np.nan) for w, _, _, ins in rep], axis=1)
        null_local.append(vals)
        null_mean.append(np.nanmean(vals, axis=1))
    thr_local = np.nanquantile(np.concatenate(null_local, axis=1), 0.95, axis=1)
    thr_mean = np.nanquantile(np.array(null_mean), 0.95, axis=0)
    w = np.concatenate([np.where(ins, w, np.nan) for w, _, _, ins in real], axis=1)
    ang = np.concatenate([a for _, a, _, _ in real], axis=1)
    sig = w > thr_local[:, None]
    frac = np.nansum(sig, axis=1) / np.sum(np.isfinite(w), axis=1)
    phase = np.angle(np.nansum(np.where(sig, np.exp(1j * ang), 0), axis=1))
    return pd.DataFrame({"period_min": period, "coherence": np.nanmean(w, axis=1), "threshold95": thr_mean,
                         "significant": np.nanmean(w, axis=1) > thr_mean, "frac_sig_area": frac,
                         "phase_deg": np.degrees(phase), "lead_min_A_over_B": phase / (2 * np.pi) * period,
                         "n_sig_points": sig.sum(axis=1)})


def ar1_surrogate(x, n, rng):
    g = max(min(np.corrcoef(x[:-1], x[1:])[0, 1], 0.95), 0.0) if len(x) > 2 else 0.0
    e = rng.normal(size=n)
    out = np.empty(n)
    out[0] = e[0]
    for t in range(1, n):
        out[t] = g * out[t - 1] + np.sqrt(1 - g * g) * e[t]
    return out


def lagged_corr(segs, max_lag=5):
    """corr(r_a(t), r_b(t + k)) รวมทุกช่วง — k > 0 สูง = A นำ B"""
    rows = []
    for k in range(-max_lag, max_lag + 1):
        xs, ys = [], []
        for x, y in segs:
            if k >= 0:
                xs.append(x[:len(x) - k]), ys.append(y[k:])
            else:
                xs.append(x[-k:]), ys.append(y[:len(y) + k])
        rows.append({"lag": k, "corr": np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1]})
    return pd.DataFrame(rows)


def coherence_map(ax, x, y, dt, s0, J, t_minutes, title, mc):
    S1, S2, S12, W12, period, coi, _ = smoothed(x, y, dt, s0, J)
    wct = np.abs(S12) ** 2 / (S1 * S2)
    # เกณฑ์ 95% ต่อคาบ (Grinsted 2004): coherence ของคู่ AR(1) อิสระ ค่าที่ 95% ของทุกเวลาในกรวย COI
    # (เขียนเองแทน pycwt.wct_significance ซึ่งเรียก numpy.randn ที่ไม่มีใน numpy 2)
    rng = np.random.default_rng(1)
    null = []
    for _ in range(mc):
        T1, T2, T12, _, _, _, ins = smoothed(ar1_surrogate(x, len(x), rng), ar1_surrogate(y, len(y), rng), dt, s0, J)
        null.append(np.where(ins, np.abs(T12) ** 2 / (T1 * T2), np.nan))
    sig = np.nanquantile(np.concatenate(null, axis=1), 0.95, axis=1)
    lp = np.log2(period)
    ax.contourf(t_minutes, lp, wct, levels=np.linspace(0, 1, 11), cmap="viridis", extend="both")
    ax.contour(t_minutes, lp, wct / sig[:, None], [1], colors="white", linewidths=1.2)
    ax.fill_between(t_minutes, np.log2(np.maximum(coi, period.min())), lp.max(), color="white", alpha=0.45,
                    hatch="xx", edgecolor="0.6", linewidth=0)
    ang = np.angle(W12)
    ti, si = slice(None, None, max(1, len(t_minutes) // 18)), slice(None, None, 4)
    ax.quiver(t_minutes[ti], lp[si], np.cos(ang[si, ti]), np.sin(ang[si, ti]), (wct[si, ti] > 0.5) * 1.0,
              cmap="Greys", clim=(0, 1), scale=30, width=0.004, headwidth=4, pivot="mid")
    yt = [p for p in [2, 5, 10, 20, 30, 60] if period.min() <= p <= period.max()]
    ax.set_yticks(np.log2(yt), [str(p) for p in yt])
    ax.set_ylim(lp.max(), lp.min())
    ax.set_title(title, fontsize=9)
    return wct, sig


def a1(mc):
    out = OUT / "a1"
    out.mkdir(parents=True, exist_ok=True)
    rows, lag_rows = [], []
    rng = np.random.default_rng(0)
    for iv in ["1m", "5m"]:
        dt = int(iv[:-1])
        for A, B in PAIRS:
            segs_all = segments(A, B, iv)
            for sess in SESS:
                segs = [(x, y) for d, s, x, y in segs_all if s == sess]
                if not segs:
                    continue
                n = min(len(x) for x, _ in segs)
                segs = [(x[:n], y[:n]) for x, y in segs]
                s0, J = scales_for(n, dt)
                d = summarize_pair(segs, dt, s0, J, mc, rng)
                d.insert(0, "n_segments", len(segs))
                d.insert(0, "session", sess)
                d.insert(0, "pair", f"{NAME.get(A, A)}–{NAME.get(B, B)}")
                d.insert(0, "interval", iv)
                rows.append(d)
                period = d["period_min"].to_numpy()
                lc = lagged_corr(segs)
                lc.insert(0, "session", sess)
                lc.insert(0, "pair", f"{NAME.get(A, A)}–{NAME.get(B, B)}")
                lc.insert(0, "interval", iv)
                lag_rows.append(lc)
                print(f"  A1 {iv} {A}–{B} {sess}: {len(segs)} ช่วง, คาบ {period.min():.0f}–{period.max():.0f} นาที")
    summ = pd.concat(rows, ignore_index=True)
    summ.to_csv(out / "coherence_by_period.csv", index=False, encoding="utf-8-sig")
    lags = pd.concat(lag_rows, ignore_index=True)
    lags.to_csv(out / "lagged_corr.csv", index=False, encoding="utf-8-sig")
    plot_summary(summ, out)
    maps_1m(out, mc)
    return summ, lags


def plot_summary(summ, out):
    pairs = list(summ["pair"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex="col")
    for i, iv in enumerate(["1m", "5m"]):
        for j, pair in enumerate(pairs):
            for k, (col_y, ylab) in enumerate([("coherence", "coherence เฉลี่ย"),
                                               ("frac_sig_area", "สัดส่วนพื้นที่มีนัยสำคัญ")]):
                ax = axes[2 * i + k, j]
                for sess, col in [("morning", "#2a78d6"), ("afternoon", "#c2410c")]:
                    d = summ[(summ.interval == iv) & (summ.pair == pair) & (summ.session == sess)]
                    if d.empty:
                        continue
                    ax.plot(d.period_min, d[col_y], color=col, lw=2, label=f"{sess} (n={d.n_segments.iat[0]})")
                    if col_y == "coherence":
                        ax.plot(d.period_min, d.threshold95, color=col, lw=1, ls="--")
                if col_y == "frac_sig_area":
                    ax.axhline(0.05, color="0.4", lw=1, ls=":")
                    ax.set_ylim(0, 0.5)
                else:
                    ax.set_ylim(0, 1)
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
                ax.set_title(f"{pair} · {iv}", fontsize=9)
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=9)
                if k == 0:
                    ax.legend(fontsize=7, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("คาบ (นาที)")
    fig.suptitle("A1: coherence เฉลี่ยตามคาบ (เส้นทึบ; เส้นประ = เกณฑ์ 95% จาก AR(1) red noise) และสัดส่วนพื้นที่ที่ "
                 "coherence เกินเกณฑ์เฉพาะจุด (เส้นจุด = 5% ที่คาดจากความบังเอิญ)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "coherence_by_period.png", dpi=150)
    plt.close(fig)


def maps_1m(out, mc):
    """แผนที่ coherence รายช่วงของ 1m (ทุกวันที่มี) ต่อคู่ ต่อรอบ"""
    for A, B in PAIRS:
        segs = segments(A, B, "1m")
        for sess in SESS:
            ss = [s for s in segs if s[1] == sess]
            if not ss:
                continue
            cols = 3
            rows = int(np.ceil(len(ss) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), squeeze=False)
            for k, (day, _, x, y) in enumerate(ss):
                n = len(x)
                s0, J = scales_for(n, 1)
                coherence_map(axes[k // cols, k % cols], x, y, 1, s0, J, np.arange(n), f"{day:%d/%m}", mc)
            for k in range(len(ss), rows * cols):
                axes[k // cols, k % cols].axis("off")
            for ax in axes[:, 0]:
                ax.set_ylabel("คาบ (นาที)")
            fig.suptitle(f"{NAME.get(A, A)}–{NAME.get(B, B)} · 1m · {sess} — สี = coherence, เส้นขาว = นัยสำคัญ 95% "
                         f"(AR1), ลายขีด = นอก COI, ลูกศร → = เฟสเดียวกัน, ↓ = {NAME.get(A, A)} นำ 90°", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"wtc_1m_{NAME.get(A, A)}_{NAME.get(B, B)}_{sess}.png", dpi=120)
            plt.close(fig)


def a2():
    """วันเดียวกัน: power spectrum (CWT) ของผลตอบแทน 1m กับ 5m + แท่ง 60m"""
    out = OUT / "a2"
    out.mkdir(parents=True, exist_ok=True)
    m1, m5, m60 = load("KBANK", "1m"), load("KBANK", "5m"), load("KBANK", "60m")
    days = sorted(set(m1.index.normalize()) & set(m5.index.normalize()))
    day = days[len(days) // 2]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1, 1.3, 1.3]})
    for j, (sess, (t0, t1)) in enumerate(SESS.items()):
        a, b = day + pd.Timedelta(t0 + ":00"), day + pd.Timedelta(t1 + ":00")
        p1, p5 = m1[a:b - pd.Timedelta("1min")], m5[a:b - pd.Timedelta("5min")]
        p60 = m60[a - pd.Timedelta("1min"):b - pd.Timedelta("1min")]
        ax = axes[0, j]
        ax.plot(p1.index, p1, lw=1, label="1m")
        ax.step(p5.index, p5, where="post", lw=1.5, label="5m")
        ax.step(p60.index, p60, where="post", lw=2.5, label="60m")
        ax.set_title(f"KBANK {day:%d/%m/%Y} {sess}: ราคาปิดแต่ละแท่ง", fontsize=10)
        ax.legend(fontsize=8)
        for i, (p, dt) in enumerate([(p1, 1), (p5, 5)]):
            r = np.diff(np.log(p.to_numpy()))
            r = (r - r.mean()) / r.std()
            W, sj, freq, coi, _, _ = pycwt.cwt(r, dt, DJ, 2 * dt, int(np.log2(150 / 3 / (2 * dt)) / DJ), MOTHER)
            period = 1 / freq
            ax = axes[i + 1, j]
            t = np.arange(len(r)) * dt
            ax.contourf(t, np.log2(period), np.log2(np.abs(W) ** 2 + 1e-6), 20, cmap="magma")
            ax.fill_between(t, np.log2(np.maximum(coi, period.min())), np.log2(period.max()), color="white",
                            alpha=0.45, hatch="xx", edgecolor="0.6", linewidth=0)
            yt = [2, 5, 10, 20, 50]
            ax.set_yticks(np.log2(yt), [str(v) for v in yt])
            ax.set_ylim(np.log2(50), np.log2(2))
            ax.axhspan(np.log2(2), np.log2(2 * dt), color="0.85")
            ax.set_title(f"scalogram {dt}m (Nyquist = {2 * dt} นาที: คาบสั้นกว่านี้มองไม่เห็น — แถบเทา)", fontsize=9)
            ax.set_ylabel("คาบ (นาที)")
            ax.set_xlabel("นาทีนับจากเริ่มรอบ")
    fig.suptitle("A2: ความละเอียดของข้อมูลกำหนดคาบสั้นสุดที่เห็นได้ — แท่ง 60m (Nyquist 120 นาที) "
                 "มีแค่ 2–3 แท่งต่อรอบ จึงไม่เห็นความเคลื่อนไหวภายในรอบเลย", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "resolution_compare.png", dpi=140)
    plt.close(fig)
    return day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=200, help="จำนวน Monte Carlo ของเกณฑ์นัยสำคัญ")
    args = ap.parse_args()
    summ, lags = a1(args.mc)
    day = a2()
    bands = summ.assign(band=pd.cut(summ.period_min, [0, 5, 10, 20, 60], labels=["<5", "5–10", "10–20", "20–60"]))
    print(bands.groupby(["interval", "pair", "session", "band"], observed=True).agg(
        coh=("coherence", "mean"), thr=("threshold95", "mean"), frac=("frac_sig_area", "mean"),
        lead=("lead_min_A_over_B", "mean")).round(3).to_string())
    best = lags.loc[lags.groupby(["interval", "pair", "session"])["corr"].idxmax()]
    print(best.round(3).to_string(index=False))
    print(f"A2 ใช้วัน {day:%d/%m/%Y} → {OUT}")


if __name__ == "__main__":
    main()
