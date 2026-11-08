"""Build a 1-minute scalogram dataset that follows the submission rules.

One sample per symbol, session and day:
    window  the last --window 1m bars that ended at or before the session cutoff
            (morning 08:30, afternoon 12:35), so bars may come from earlier days
    label   1 if the session end price is above the session start price, otherwise 0 (flat = not up)
            morning: P(12:30) vs P(10:00) · afternoon: official close after ATC vs P(14:00)
    prices  taken from the project dataset, the same labels used for scoring

--step and --horizon are accepted so older commands still run, but they have no effect.
Output: outputs/demo/scalogram_1m_dataset/<session>/

Usage examples:
    python experiments/25_1m_scalogram_demo.py --symbols KBANK ADVANC --all-days --session morning --window 80
    python experiments/25_1m_scalogram_demo.py --symbols KBANK ADVANC --all-days --session afternoon --window 80
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _setup  # noqa: F401
from forecast.config import OUTPUT_DIR, ROOT, load_rules, session_times
from forecast.data.io import frame_from_raw
from forecast.evaluation.runner import load_or_build_dataset
from forecast.features.images import _resize, _unit, morlet_cwt

TZ = "Asia/Bangkok"


def cwt_image_from_bars(bars):
    """Create a 3-channel Morlet scalogram image from a minute-bar window."""
    c = np.log(bars["close"].to_numpy(float))
    r = np.diff(c, prepend=c[0]) * 100.0
    v = np.log1p(bars["volume"].to_numpy(float))
    v = (v - v.mean()) / (v.std() + 1e-8)
    series = np.stack([r, np.abs(r), v], axis=0)
    chans = []
    for ch in series:
        z = (ch - ch.mean()) / (ch.std() + 1e-8)
        w = np.log1p(np.abs(morlet_cwt(z, n_scales=32)))
        chans.append(_unit(w))
    img = _resize(np.stack(chans, axis=0).astype(np.float32), 224, 224)
    return np.clip(img, 0, 1)


def window_at_cutoff(bars, cutoff, window_size):
    """Last window_size bars that ended at or before cutoff, or None if there are too few."""
    known = bars[bars["bar_end"] <= cutoff].sort_values("bar_end")
    return known.iloc[-window_size:] if len(known) >= window_size else None


def session_labels(rules, session):
    """Scored rows of one session: start price, reference price, end price and direction."""
    assert rules["label"]["price_1630"] == "closing_auction" and rules["label"]["p_ref_morning"] == "official_close"
    ds = load_or_build_dataset(rules, tag="_new")
    rows = ds[ds["session"] == session]
    return rows[rows[["p_start", "p_ref", "p_end", "direction"]].notna().all(axis=1)]


def build_dataset_for_symbol(symbol, bars, labels, rules, session, window_size, dates=None):
    """→ (samples, days skipped because the window was too short)"""
    rows, skipped = [], []
    for r in labels[labels["symbol"] == symbol].sort_values("date").itertuples():
        day_str = str(r.date.date())
        if dates is not None and day_str not in dates:
            continue
        cutoff = session_times(rules, session, r.date)["cutoff"]
        win = window_at_cutoff(bars, cutoff, window_size)
        if win is None:
            skipped.append(day_str)
            continue
        rows.append({
            "symbol": symbol,
            "date": day_str,
            "session": session,
            "label": int(r.direction == 1),
            "window_end": str(win["bar_end"].iloc[-1]),
            "p_start": float(r.p_start),
            "price_end": float(r.p_ref),
            "session_close": float(r.p_end),
            "image": cwt_image_from_bars(win),
        })
    return rows, skipped


def save_png(img, out_path):
    plt.imsave(out_path, img.transpose(1, 2, 0))


def split_dates(dates, train_ratio=0.7, valid_ratio=0.15):
    if len(dates) < 3:
        return {"train": dates, "valid": [], "test": []}
    train_end = int(len(dates) * train_ratio)
    valid_end = train_end + int(len(dates) * valid_ratio)
    return {"train": dates[:train_end], "valid": dates[train_end:valid_end], "test": dates[valid_end:]}


def stack_split(df_split):
    if df_split.empty:
        return np.empty((0, 3, 224, 224), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(df_split["image"].to_numpy()), df_split["label"].to_numpy(dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["KBANK", "ADVANC"])
    ap.add_argument("--date", help="Example: 2026-10-19")
    ap.add_argument("--session", choices=["morning", "afternoon"], default="morning")
    ap.add_argument("--all-days", action="store_true")
    ap.add_argument("--window", type=int, default=80, help="Number of minute bars per sample")
    ap.add_argument("--step", type=int, help="No effect: each symbol has one sample per session and day")
    ap.add_argument("--horizon", type=int, help="No effect: the label comes from the session end price")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--valid-ratio", type=float, default=0.15)
    args = ap.parse_args()
    if args.step is not None or args.horizon is not None:
        print("note: --step and --horizon have no effect in this version")

    rules = load_rules()
    labels = session_labels(rules, args.session)
    out_dir = OUTPUT_DIR / "demo" / "scalogram_1m_dataset" / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = None
    if args.date:
        dates = {args.date}
    elif not args.all_days:
        dates = {str(labels["date"].max().date())}

    all_rows = []
    for symbol in args.symbols:
        path = ROOT / "data" / "raw" / "settrade" / f"{symbol}_1m.csv"
        if not path.exists():
            print(f"skip {symbol}: missing raw file")
            continue
        bars = frame_from_raw(pd.read_csv(path), rules, "1m")
        first = bars["bar_end"].min().tz_convert(TZ).normalize()
        rows, skipped = build_dataset_for_symbol(symbol, bars, labels[labels["date"] >= first], rules, args.session, args.window, dates)
        all_rows.extend(rows)
        if skipped:
            print(f"{symbol}: skipped {skipped}: fewer than {args.window} bars before the cutoff")
        for row in rows:
            save_png(row["image"], out_dir / f"{symbol}_{row['date']}_{args.session}_sample.png")
        print(f"{symbol} {args.session}: {len(rows)} samples")

    if not all_rows:
        raise SystemExit("No valid 1m scalogram samples were created; check the raw data or parameter values.")

    df = pd.DataFrame(all_rows)
    days = sorted(df["date"].unique())
    if len(days) < 3:
        # Fallback for a single-day demo: split by sample index instead of date.
        n = len(df)
        train_end = max(1, int(n * args.train_ratio))
        valid_end = max(train_end + 1, min(n, train_end + int(n * args.valid_ratio)))
        splits = {"train": df.iloc[:train_end], "valid": df.iloc[train_end:valid_end], "test": df.iloc[valid_end:]}
    else:
        splits = {k: df[df["date"].isin(v)] for k, v in split_dates(days, args.train_ratio, args.valid_ratio).items()}

    columns = ["symbol", "date", "session", "label", "window_end", "p_start", "price_end", "session_close"]
    for key in ["train", "valid", "test"]:
        X, y = stack_split(splits[key])
        close_target = splits[key]["session_close"].to_numpy(dtype=np.float32)
        np.savez_compressed(out_dir / f"{key}.npz", X=X, y=y, close=close_target)
        splits[key][columns].to_csv(out_dir / f"{key}.csv", index=False)
        print(f"{key}: X={X.shape}, y={y.shape}, close={close_target.shape}, balance={y.mean() if len(y) else 0.0:.3f}")

    pd.DataFrame([
        {"split": k, "n_samples": len(v), "n_pos": int(v["label"].sum()), "n_neg": int((1 - v["label"]).sum())}
        for k, v in splits.items()
    ]).to_csv(out_dir / "split_summary.csv", index=False)

    print(f"dataset saved to: {out_dir}")


if __name__ == "__main__":
    main()
