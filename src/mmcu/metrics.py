"""Evaluation metrics. AUROC is the Hateful Memes headline metric; the rest are
what a trust/safety team actually needs: per-class precision/recall, a confusion
matrix at an explicit operating threshold, and calibration (ECE)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of P(hateful), equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing macro-F1; chosen on validation, then frozen for test."""
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, p >= t, average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(y, yhat, labels=[0, 1], zero_division=0)
    two_class = len(np.unique(y)) == 2
    return {
        "auroc": float(roc_auc_score(y, p)) if two_class else float("nan"),
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(), "support": sup.tolist(),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
        "ece": ece(y, p),
        "threshold": float(threshold),
        "n": int(len(y)),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn=roc_auc_score, n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 2000,
                           seed: int = 0) -> dict:
    """AUROC(a) - AUROC(b) on the same resamples; `p_value` is the one-sided P(delta <= 0)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    d = np.array(d)
    return {"delta": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_value": float((d <= 0).mean())}
