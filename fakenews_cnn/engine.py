"""Training / inference loops."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False,
                    grad_clip=0.0, desc="train"):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        n += bs
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def infer(model, loader, device, criterion=None, use_amp=False, desc="eval"):
    """Returns (labels, softmax probs, mean loss or None)."""
    model.eval()
    all_labels, all_probs = [], []
    total_loss, n = 0.0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, labels_d).item() * labels_d.size(0)
        n += labels_d.size(0)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    loss = total_loss / max(n, 1) if criterion is not None else None
    return np.concatenate(all_labels), np.concatenate(all_probs), loss
