"""Metric computation (computed from real predictions only)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

from .config import CLASS_NAMES


def compute_metrics(y_true, probs, pos_index: int = 0) -> dict:
    """probs: (N, 2) softmax probabilities. Precision/recall/F1/ROC-AUC are for the positive class."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)

    yt_bin = (y_true == pos_index).astype(int)
    yp_bin = (y_pred == pos_index).astype(int)
    score = probs[:, pos_index]

    auc = None
    if len(np.unique(yt_bin)) == 2:
        auc = float(roc_auc_score(yt_bin, score))

    return {
        "positive_class": CLASS_NAMES[pos_index],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(yt_bin, yp_bin, zero_division=0)),
        "recall": float(recall_score(yt_bin, yp_bin, zero_division=0)),
        "f1": float(f1_score(yt_bin, yp_bin, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": auc,
    }
