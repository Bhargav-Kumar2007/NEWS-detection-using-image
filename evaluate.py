#!/usr/bin/env python
"""Evaluate a trained checkpoint: metrics, classification report, confusion matrix, ROC curve.

Usage
  python evaluate.py                         # best checkpoint of the current config, test split
  python evaluate.py --split val
  python evaluate.py --checkpoint path/to/best.pt
  python evaluate.py --full                  # use the checkpoint trained on the full dataset
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from fakenews_cnn.config import (CLASS_NAMES, add_common_args, apply_common_overrides, load_config,
                                 positive_index, resolve, run_paths)
from fakenews_cnn.data import make_loader, manifest_path
from fakenews_cnn.engine import infer
from fakenews_cnn.metrics import compute_metrics
from fakenews_cnn.models import build_model
from fakenews_cnn.plotting import plot_confusion_matrix, plot_roc_curve
from fakenews_cnn.utils import describe_device, get_device, get_logger, save_json


def load_checkpoint_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def evaluate_checkpoint(cfg: dict, checkpoint, split: str = "test", device=None,
                        out_dir=None, logger=None) -> dict:
    log = logger or get_logger("evaluate")
    device = device or get_device(cfg.get("device", "auto"))
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run train.py first.")
    if not manifest_path(cfg, split).exists():
        raise FileNotFoundError(f"Manifest for split '{split}' not found. Run download_data.py first.")

    model, ckpt = load_checkpoint_model(checkpoint, device)
    paths = run_paths(cfg, create=True)
    metrics_dir = Path(out_dir) if out_dir else paths["metrics"]
    plots_dir = Path(out_dir) if out_dir else paths["plots"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pos = positive_index(cfg)
    loader = make_loader(cfg, split, train=False)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    log.info(f"Evaluating {checkpoint.name} on '{split}' split ({len(loader.dataset)} samples) "
             f"on {describe_device(device)}")
    labels, probs, _ = infer(model, loader, device, use_amp=use_amp, desc=f"eval-{split}")
    preds = probs.argmax(1)

    m = compute_metrics(labels, probs, pos)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    report_txt = classification_report(labels, preds, labels=[0, 1], target_names=CLASS_NAMES,
                                       digits=4, zero_division=0)
    report_dict = classification_report(labels, preds, labels=[0, 1], target_names=CLASS_NAMES,
                                        digits=4, zero_division=0, output_dict=True)

    result = {
        "split": split,
        "n_samples": int(len(labels)),
        "checkpoint": str(checkpoint),
        "arch": ckpt["arch"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": m,
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": cm.tolist()},
        "classification_report": report_dict,
    }
    save_json(result, metrics_dir / f"metrics_{split}.json")
    (metrics_dir / f"classification_report_{split}.txt").write_text(report_txt, encoding="utf-8")
    plot_confusion_matrix(cm, CLASS_NAMES, plots_dir / f"confusion_matrix_{split}.png",
                          title=f"Confusion Matrix - {split}")
    roc_ok = plot_roc_curve((labels == pos).astype(int), probs[:, pos],
                            plots_dir / f"roc_curve_{split}.png", CLASS_NAMES[pos])

    ids = pd.read_csv(manifest_path(cfg, split), dtype={"id": str})["id"].tolist()
    pd.DataFrame({"id": ids, "true_label": labels, "pred_label": preds,
                  "prob_fake": probs[:, 0], "prob_real": probs[:, 1]}
                 ).to_csv(metrics_dir / f"predictions_{split}.csv", index=False)

    auc_txt = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a (single class in split)"
    log.info(f"[{split}] acc={m['accuracy']:.4f} precision={m['precision']:.4f} "
             f"recall={m['recall']:.4f} f1={m['f1']:.4f} roc_auc={auc_txt} "
             f"(positive class: {m['positive_class']})")
    log.info("\n" + report_txt)
    if not roc_ok:
        log.warning("ROC curve skipped: only one class present in this split.")
    log.info(f"Saved metrics to {metrics_dir} and plots to {plots_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: run's best.pt)")
    parser.add_argument("--output-dir", default=None, help="Write all outputs into this folder")
    args = parser.parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    ckpt = resolve(args.checkpoint) if args.checkpoint else run_paths(cfg)["checkpoints"] / "best.pt"
    evaluate_checkpoint(cfg, ckpt, args.split, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
