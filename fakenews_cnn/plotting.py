"""Matplotlib visualisations (all drawn from real run data)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return
    specs = [("accuracy", "train_acc", "val_acc", "Accuracy"),
             ("loss", "train_loss", "val_loss", "Loss")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (_, tk, vk, title) in zip(axes, specs):
        ax.plot(epochs, history[tk], marker="o", label="Train")
        ax.plot(epochs, history[vk], marker="o", label="Validation")
        ax.set_title(f"Training vs Validation {title}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.grid(alpha=0.3); ax.legend()
    _save(fig, out_dir / "training_curves.png")

    for name, tk, vk, title in specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(epochs, history[tk], marker="o", label="Train")
        ax.plot(epochs, history[vk], marker="o", label="Validation")
        ax.set_title(f"Training vs Validation {title}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.grid(alpha=0.3); ax.legend()
        _save(fig, out_dir / f"{name}_curve.png")


def plot_confusion_matrix(cm, class_names, path, title="Confusion Matrix") -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Counts"), (axes[1], norm, ".2%", "Row-normalised")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{title} ({sub})")
        thresh = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_roc_curve(y_true_bin, scores, path, positive_name="fake") -> bool:
    y_true_bin = np.asarray(y_true_bin)
    if len(np.unique(y_true_bin)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (positive class: {positive_name})")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _save(fig, path)
    return True
