#!/usr/bin/env python
"""Train a pretrained ResNet18 / MobileNetV2 on Fakeddit (2-way fake vs real, images only).

Usage
  python train.py                       # small subset, settings from config.yaml
  python train.py --full                # full official dataset
  python train.py --model mobilenet_v2 --epochs 5 --batch-size 64
  python train.py --resume              # continue from outputs/<run>/checkpoints/last.pt

If the dataset has not been prepared yet, it is downloaded automatically first.
Everything (checkpoints, history, metrics, plots, logs) goes to outputs/<subset|full>_<arch>/.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from download_data import manifests_exist, prepare_data
from evaluate import evaluate_checkpoint
from fakenews_cnn.config import (CLASS_NAMES, add_common_args, apply_common_overrides,
                                 dataset_mode, load_config, positive_index, run_paths)
from fakenews_cnn.data import make_loader, manifest_path
from fakenews_cnn.engine import infer, train_one_epoch
from fakenews_cnn.metrics import compute_metrics
from fakenews_cnn.models import build_model
from fakenews_cnn.plotting import plot_history
from fakenews_cnn.utils import (describe_device, get_device, get_logger, save_json, seed_everything)

MONITOR_MODES = {"val_f1": "max", "val_acc": "max", "val_auc": "max", "val_loss": "min"}


def save_checkpoint(path, model, cfg, epoch, val_metrics, extra=None):
    payload = {
        "model_state": model.state_dict(),
        "arch": cfg["model"]["arch"],
        "class_names": CLASS_NAMES,
        "image_size": int(cfg["data"]["image_size"]),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config": cfg,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt of this run")
    parser.add_argument("--no-eval", action="store_true", help="Skip test evaluation after training")
    args = parser.parse_args()

    cfg = apply_common_overrides(load_config(args.config), args)
    t = cfg["train"]
    if args.epochs is not None:
        t["epochs"] = args.epochs
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size
    if args.lr is not None:
        t["lr"] = args.lr

    paths = run_paths(cfg, create=True)
    log = get_logger("train", paths["base"] / "train.log")
    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    use_amp = bool(t.get("amp", True)) and device.type == "cuda"
    log.info(f"Device: {describe_device(device)} | AMP: {use_amp} | mode: {dataset_mode(cfg)} "
             f"| model: {cfg['model']['arch']}")

    if not manifests_exist(cfg):
        log.info("Dataset not prepared yet - running download_data.py steps automatically.")
        prepare_data(cfg)

    train_loader = make_loader(cfg, "train", train=True)
    val_loader = make_loader(cfg, "val", train=False)
    log.info(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = build_model(cfg["model"]["arch"], pretrained=bool(cfg["model"]["pretrained"]),
                        num_classes=2, dropout=float(cfg["model"]["dropout"]),
                        freeze_backbone=bool(cfg["model"].get("freeze_backbone", False))).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_train:,}")

    weight = None
    if t.get("class_weights", True):
        counts = np.bincount(train_loader.dataset.labels, minlength=2).astype(float)
        if (counts > 0).all():
            weight = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
            log.info(f"Class weights (fake, real): {weight.cpu().numpy().round(4).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(t.get("label_smoothing", 0.0)))
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    epochs = int(t["epochs"])
    sched_name = str(t.get("scheduler", "cosine")).lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    else:
        scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    monitor = t.get("monitor", "val_f1")
    if monitor not in MONITOR_MODES:
        raise ValueError(f"train.monitor must be one of {list(MONITOR_MODES)}")
    sign = 1.0 if MONITOR_MODES[monitor] == "max" else -1.0
    best_score, best_epoch, bad_epochs, start_epoch = -np.inf, 0, 0, 1
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_precision",
                               "val_recall", "val_f1", "val_auc", "lr", "epoch_seconds"]}

    last_path = paths["checkpoints"] / "last.pt"
    best_path = paths["checkpoints"] / "best.pt"
    if args.resume:
        if last_path.exists():
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            if scheduler is not None and ck.get("scheduler_state"):
                scheduler.load_state_dict(ck["scheduler_state"])
            if ck.get("scaler_state"):
                scaler.load_state_dict(ck["scaler_state"])
            history, best_score = ck["history"], ck["best_score"]
            best_epoch, bad_epochs, start_epoch = ck["best_epoch"], ck["bad_epochs"], ck["epoch"] + 1
            log.info(f"Resumed from epoch {ck['epoch']}")
        else:
            log.warning("--resume given but no last.pt found; starting from scratch.")

    pos = positive_index(cfg)
    save_json(cfg, paths["base"] / "config_used.json")
    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                          use_amp, float(t.get("grad_clip", 0.0)),
                                          desc=f"epoch {epoch}/{epochs} train")
        labels, probs, val_loss = infer(model, val_loader, device, val_criterion, use_amp,
                                        desc=f"epoch {epoch}/{epochs} val")
        vm = compute_metrics(labels, probs, pos)
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss) if sched_name == "plateau" else scheduler.step()

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc), ("val_loss", val_loss),
                     ("val_acc", vm["accuracy"]), ("val_precision", vm["precision"]),
                     ("val_recall", vm["recall"]), ("val_f1", vm["f1"]),
                     ("val_auc", vm["roc_auc"] if vm["roc_auc"] is not None else float("nan")),
                     ("lr", lr_now), ("epoch_seconds", time.time() - ep_start)]:
            history[k].append(float(v))

        current = {"val_f1": vm["f1"], "val_acc": vm["accuracy"],
                   "val_auc": vm["roc_auc"] if vm["roc_auc"] is not None else -np.inf,
                   "val_loss": val_loss}[monitor]
        score = sign * current
        improved = score > best_score
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(best_path, model, cfg, epoch, {**vm, "loss": val_loss})
        else:
            bad_epochs += 1

        auc_txt = f"{vm['roc_auc']:.4f}" if vm["roc_auc"] is not None else "n/a"
        log.info(f"Epoch {epoch}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                 f"val loss {val_loss:.4f} acc {vm['accuracy']:.4f} f1 {vm['f1']:.4f} auc {auc_txt}"
                 f"{'  <- best' if improved else ''}")

        save_checkpoint(last_path, model, cfg, epoch, {**vm, "loss": val_loss}, extra={
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if use_amp else None,
            "history": history, "best_score": best_score, "best_epoch": best_epoch,
            "bad_epochs": bad_epochs})
        save_json(history, paths["metrics"] / "history.json")
        pd.DataFrame(history).assign(epoch=range(1, len(history["train_loss"]) + 1)
                                     ).to_csv(paths["metrics"] / "history.csv", index=False)
        plot_history(history, paths["plots"])

        patience = int(t.get("early_stopping_patience", 0))
        if patience and bad_epochs >= patience:
            log.info(f"Early stopping: no improvement in '{monitor}' for {patience} epochs.")
            break

    total = time.time() - t0
    save_json({"best_epoch": best_epoch, "monitor": monitor, "best_monitor_value": sign * best_score,
               "epochs_run": len(history["train_loss"]), "training_seconds": total,
               "device": describe_device(device), "best_checkpoint": str(best_path)},
              paths["metrics"] / "training_summary.json")
    log.info(f"Training finished in {total / 60:.1f} min. Best epoch: {best_epoch} "
             f"({monitor} = {sign * best_score:.4f}). Best checkpoint: {best_path}")

    if t.get("evaluate_after", True) and not args.no_eval and manifest_path(cfg, "test").exists():
        for split in ("val", "test"):
            evaluate_checkpoint(cfg, best_path, split, device=device, logger=log)
    log.info(f"All outputs saved in {paths['base']}")


if __name__ == "__main__":
    main()
