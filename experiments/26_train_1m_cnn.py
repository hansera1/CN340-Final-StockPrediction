"""Train two models for the 1-minute scalogram dataset.

Model A: classification head for up/down prediction
Model B: regression head for session close price, using A's predicted probability as a feature
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNetClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model = model
        self.head = nn.Linear(model.fc.in_features, 1)
        model.fc = nn.Identity()

    def forward(self, x):
        features = self.model(x)
        return self.head(features).squeeze(1)


def normalize_image_batch(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def load_npz(path: Path):
    obj = np.load(path)
    X = obj["X"]
    y = obj["y"].astype(np.float32)
    close = obj["close"].astype(np.float32) if "close" in obj else np.zeros_like(y, dtype=np.float32)
    if X.ndim != 4:
        raise ValueError(f"Expected 4D image tensor in {path}, got shape {X.shape}")
    if X.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {X.shape}")
    return X, y, close


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalize_image_batch(xb))
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
        total += yb.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits_all = []
    labels_all = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(normalize_image_batch(xb))
        logits_all.append(logits.cpu())
        labels_all.append(yb.cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return logits.numpy(), labels.numpy()


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        torch.from_numpy(np.ascontiguousarray(y)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def wf_plan(days, min_train_days):
    """(training days, test day) for every day that has at least min_train_days before it."""
    return [(days[:i], days[i]) for i in range(min_train_days, len(days))]


def fit_classifier(X, y, idx_train, idx_valid, epochs, args, device, seed, pretrained=True):
    """Train on idx_train. With idx_valid, keep the best epoch by its loss. → (model, best epoch)"""
    torch.manual_seed(seed)
    model = ResNetClassifier(pretrained=pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
    loader = make_loader(X[idx_train], y[idx_train], args.batch_size, shuffle=True)
    best_state, best_loss, best_epoch = None, float("inf"), epochs
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        if len(idx_valid):
            logits, labels = predict_logits(model, make_loader(X[idx_valid], y[idx_valid], args.batch_size, shuffle=False), device)
            loss = criterion(torch.tensor(logits), torch.tensor(labels)).item()
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def predict_prob(model, X, y, idx, args, device):
    logits, _ = predict_logits(model, make_loader(X[idx], y[idx], args.batch_size, shuffle=False), device)
    return 1.0 / (1.0 + np.exp(-logits))


def walk_forward(X, y, meta, min_train_days, args, device, seed=0, pretrained=True):
    """Each test day uses only earlier days.

    Model A: trained on all earlier days except the last, which picks the best epoch.
    Model B: price_end + probability, where each training day's probability comes from
    a Model A that did not train on that day (out-of-fold).
    """
    dates = meta["date"].to_numpy()
    rows = []
    for train_days, test_day in wf_plan(sorted(meta["date"].unique()), min_train_days):
        fit_idx = np.flatnonzero(np.isin(dates, train_days[:-1]))
        val_idx = np.flatnonzero(dates == train_days[-1])
        test_idx = np.flatnonzero(dates == test_day)
        train_idx = np.flatnonzero(np.isin(dates, train_days))
        model, best_epoch = fit_classifier(X, y, fit_idx, val_idx, args.epochs, args, device, seed, pretrained)
        test_prob = predict_prob(model, X, y, test_idx, args, device)
        oof = np.full(len(meta), np.nan, dtype=np.float32)
        for day in train_days:
            held = np.flatnonzero(dates == day)
            rest = np.flatnonzero(np.isin(dates, train_days) & (dates != day))
            fold_model, _ = fit_classifier(X, y, rest, np.empty(0, dtype=int), best_epoch, args, device, seed, pretrained)
            oof[held] = predict_prob(fold_model, X, y, held, args, device)
        reg_x = np.column_stack([meta["price_end"].to_numpy(dtype=np.float32), oof])
        model_b = LinearRegression().fit(reg_x[train_idx], meta["session_close"].to_numpy(dtype=np.float32)[train_idx])
        test_x = np.column_stack([meta["price_end"].to_numpy(dtype=np.float32)[test_idx], test_prob])
        day_rows = meta.iloc[test_idx].copy()
        day_rows["train_days"] = len(train_days)
        day_rows["best_epoch"] = best_epoch
        day_rows["pred_up_prob"] = test_prob.astype(float)
        day_rows["pred_session_close"] = model_b.predict(test_x).astype(float)
        share = meta.iloc[train_idx].groupby("symbol")["label"].mean()
        day_rows["base_up_rate"] = day_rows["symbol"].map(share).astype(float)
        rows.append(day_rows)
        print(f"{test_day}: train_days={len(train_days)} best_epoch={best_epoch}", flush=True)
    return pd.concat(rows, ignore_index=True)


def walk_forward_summary(p):
    """Compare the model with always answering not up and with the reference price P(c)."""
    y = p["label"].to_numpy(float)
    ll = lambda q: float(np.mean(-(y * np.log(np.clip(q, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(1 - np.clip(q, 1e-6, 1 - 1e-6)))))
    close = p["session_close"].to_numpy(float)
    err_b, err_pc = p["pred_session_close"].to_numpy(float) - close, p["price_end"].to_numpy(float) - close
    prob = p["pred_up_prob"].to_numpy(float)
    return {
        "n_rows": int(len(p)), "n_days": int(p["date"].nunique()), "up_rate": float(y.mean()), "pred_up_rate": float((prob >= 0.5).mean()),
        "accuracy_model_a": float(((prob >= 0.5) == (y == 1)).mean()), "accuracy_always_not_up": float((y == 0).mean()),
        "logloss_model_a": ll(prob), "logloss_train_up_rate": ll(p["base_up_rate"].to_numpy(float)),
        "rmse_model_b": float(np.sqrt(np.mean(err_b ** 2))), "rmse_reference_price": float(np.sqrt(np.mean(err_pc ** 2))),
        "mae_model_b": float(np.mean(np.abs(err_b))), "mae_reference_price": float(np.mean(np.abs(err_pc))),
    }


def run_walk_forward(out_dir, args, device):
    parts = [load_npz(out_dir / f"{k}.npz") for k in ("train", "valid", "test")]
    X, y = np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
    meta = pd.concat([pd.read_csv(out_dir / f"{k}.csv") for k in ("train", "valid", "test")], ignore_index=True)
    if not np.array_equal(meta["label"].to_numpy(dtype=np.float32), y):
        raise ValueError("csv rows do not match npz rows")
    p = walk_forward(X, y, meta, args.min_train_days, args, device, args.seed)
    p["pred_up_class"] = (p["pred_up_prob"] >= 0.5).astype(int)
    p["correct_updown"] = (p["pred_up_class"] == p["label"]).astype(int)
    p["close_error"] = p["pred_session_close"] - p["session_close"]
    p["close_abs_error"] = p["close_error"].abs()
    p["reference_error"] = p["price_end"] - p["session_close"]
    p.to_csv(out_dir / "walk_forward_predictions.csv", index=False)
    summary = walk_forward_summary(p)
    with open(out_dir / "walk_forward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"walk-forward predictions csv: {out_dir / 'walk_forward_predictions.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-forward", action="store_true", help="Train on earlier days and predict each later day")
    ap.add_argument("--min-train-days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--session", choices=["morning", "afternoon"], required=True, help="Which dataset from 25_1m_scalogram_demo.py to train on")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "demo" / "scalogram_1m_dataset" / args.session
    train_path = out_dir / "train.npz"
    valid_path = out_dir / "valid.npz"

    if args.device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    if args.walk_forward:
        run_walk_forward(out_dir, args, device)
        return

    X_train, y_train, close_train = load_npz(train_path)
    X_valid, y_valid, close_valid = load_npz(valid_path)

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    valid_loader = make_loader(X_valid, y_valid, args.batch_size, shuffle=False)

    model_a = ResNetClassifier(pretrained=True).to(device)
    criterion_a = nn.BCEWithLogitsLoss()
    optimizer_a = AdamW(model_a.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_a = StepLR(optimizer_a, step_size=5, gamma=0.7)

    best_state = None
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model_a, train_loader, optimizer_a, criterion_a, device)
        logits_valid, y_valid_true = predict_logits(model_a, valid_loader, device)
        valid_prob = 1.0 / (1.0 + np.exp(-logits_valid))
        valid_loss = criterion_a(torch.tensor(logits_valid, dtype=torch.float32, device=device), torch.tensor(y_valid_true, dtype=torch.float32, device=device)).item()
        scheduler_a.step()

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().clone() for k, v in model_a.state_dict().items()}

        acc = accuracy_score(y_valid_true, (valid_prob >= 0.5).astype(int))
        print(f"Model A epoch={epoch:02d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} acc={acc:.3f}")

    if best_state is not None:
        model_a.load_state_dict(best_state)

    # Keep row order so train_prob stays aligned with train_meta.
    train_eval_loader = make_loader(X_train, y_train, args.batch_size, shuffle=False)
    logits_train, y_train_true = predict_logits(model_a, train_eval_loader, device)
    logits_valid, y_valid_true = predict_logits(model_a, valid_loader, device)
    train_prob = 1.0 / (1.0 + np.exp(-logits_train))
    valid_prob = 1.0 / (1.0 + np.exp(-logits_valid))

    class_metrics = compute_metrics(y_valid_true, valid_prob)
    print(json.dumps({"model_a_metrics": class_metrics}, indent=2))

    train_meta = pd.read_csv(out_dir / "train.csv")
    valid_meta = pd.read_csv(out_dir / "valid.csv")
    if len(train_meta) != len(train_prob):
        raise ValueError(f"train.csv rows ({len(train_meta)}) do not match train.npz rows ({len(train_prob)})")
    if len(valid_meta) != len(valid_prob):
        raise ValueError(f"valid.csv rows ({len(valid_meta)}) do not match valid.npz rows ({len(valid_prob)})")

    train_reg_x = np.column_stack([
        train_meta["price_end"].to_numpy(dtype=np.float32),
        train_prob.astype(np.float32),
    ])
    valid_reg_x = np.column_stack([
        valid_meta["price_end"].to_numpy(dtype=np.float32),
        valid_prob.astype(np.float32),
    ])
    model_b = LinearRegression()
    model_b.fit(train_reg_x, train_meta["session_close"].to_numpy(dtype=np.float32))
    valid_close_pred = model_b.predict(valid_reg_x)
    valid_true_close = valid_meta["session_close"].to_numpy(dtype=np.float32)

    out_df = valid_meta.copy()
    out_df["pred_up_prob"] = valid_prob.astype(float)
    out_df["pred_up_class"] = (valid_prob >= 0.5).astype(int)
    out_df["true_up_label"] = out_df["label"].astype(int)
    out_df["pred_session_close"] = valid_close_pred.astype(float)
    out_df["true_session_close"] = valid_true_close.astype(float)
    out_df["correct_updown"] = (out_df["true_up_label"] == out_df["pred_up_class"]).astype(int)
    out_df["close_error"] = out_df["pred_session_close"] - out_df["true_session_close"]
    out_df["close_abs_error"] = out_df["close_error"].abs()
    out_df["close_correct_1baht"] = (out_df["close_abs_error"] <= 1.0).astype(int)
    out_df["close_correct_2baht"] = (out_df["close_abs_error"] <= 2.0).astype(int)

    out_csv = out_dir / "valid_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"valid predictions csv: {out_csv}")

    summary = (
        out_df.groupby(["symbol", "date", "session"], as_index=False)
        .agg(
            samples=("symbol", "size"),
            updown_accuracy=("correct_updown", "mean"),
            close_correct_1baht=("close_correct_1baht", "mean"),
            close_correct_2baht=("close_correct_2baht", "mean"),
            close_mae=("close_abs_error", "mean"),
            close_rmse=("close_error", lambda s: float(np.sqrt(np.mean(np.square(s)))),),
        )
    )
    summary_csv = out_dir / "valid_session_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"valid session summary csv: {summary_csv}")

    model_b_rmse = float(np.sqrt(np.mean((valid_close_pred - valid_true_close) ** 2)))
    close_correct_1baht = float(np.mean(out_df["close_correct_1baht"]))
    metrics_path = out_dir / "valid_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_a": class_metrics,
            "model_b_rmse": model_b_rmse,
            "model_b_close_correct_1baht": close_correct_1baht,
        }, f, ensure_ascii=False, indent=2)
    print(f"metrics json: {metrics_path}")
    print(f"Model B RMSE = {model_b_rmse:.4f}")
    print(f"Model B close within 1 baht = {close_correct_1baht:.3f}")


if __name__ == "__main__":
    main()
