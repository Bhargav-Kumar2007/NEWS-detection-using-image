#!/usr/bin/env python
"""Reusable prediction + feature-extraction API (also usable from the command line).

Python
  from predict import predict_image, predict_images, extract_features, Predictor

  predict_image("photo.jpg")                       # dict: label, confidence, probabilities
  predict_images(["a.jpg", "b.png"])               # list of dicts
  extract_features("photo.jpg")                    # np.ndarray (512,) ResNet18 / (1280,) MobileNetV2
  extract_features(["a.jpg", "b.jpg"])             # np.ndarray (N, D)

  p = Predictor("outputs/subset_resnet18/checkpoints/best.pt")   # reuse a loaded model
  p.predict("photo.jpg"); p.extract_features("photo.jpg")

Inputs may be file paths, http(s) URLs, PIL images or numpy arrays.

CLI
  python predict.py image1.jpg image2.png
  python predict.py img.jpg --checkpoint outputs/full_resnet18/checkpoints/best.pt
  python predict.py img1.jpg img2.jpg --features-out features.npy

NOTE: this model judges an image by patterns learned from Fakeddit's subreddit-derived labels.
It is not a fact-checker; do not treat its output as proof that an image is truthful or fake.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from fakenews_cnn.config import ROOT, resolve
from fakenews_cnn.data import eval_transform
from fakenews_cnn.models import build_model, forward_features
from fakenews_cnn.utils import get_device

_CACHE: dict = {}


def find_default_checkpoint() -> Path:
    """Most recently modified outputs/*/checkpoints/best.pt."""
    candidates = sorted((ROOT / "outputs").glob("*/checkpoints/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained checkpoint found in outputs/. Run train.py first, "
                                "or pass checkpoint=... explicitly.")
    return candidates[0]


def load_image(src) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, np.ndarray):
        return Image.fromarray(src).convert("RGB")
    s = str(src)
    if s.startswith(("http://", "https://")):
        r = requests.get(s, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(s).convert("RGB")


class Predictor:
    def __init__(self, checkpoint=None, device="auto", batch_size: int = 32):
        self.checkpoint = Path(resolve(checkpoint)) if checkpoint else find_default_checkpoint()
        self.device = get_device(device)
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.arch = ck["arch"]
        self.class_names = list(ck["class_names"])
        self.image_size = int(ck.get("image_size", 224))
        self.model = build_model(self.arch, pretrained=False, num_classes=len(self.class_names))
        self.model.load_state_dict(ck["model_state"])
        self.model.to(self.device).eval()
        self.transform = eval_transform(self.image_size)
        self.batch_size = batch_size

    def _batches(self, images):
        for i in range(0, len(images), self.batch_size):
            chunk = images[i:i + self.batch_size]
            x = torch.stack([self.transform(load_image(im)) for im in chunk]).to(self.device)
            yield x

    @torch.no_grad()
    def predict_batch(self, images) -> list[dict]:
        results = []
        for x in self._batches(list(images)):
            probs = torch.softmax(self.model(x).float(), dim=1).cpu().numpy()
            for p in probs:
                idx = int(p.argmax())
                results.append({
                    "label": self.class_names[idx],
                    "label_id": idx,
                    "confidence": float(p[idx]),
                    "probabilities": {n: float(p[i]) for i, n in enumerate(self.class_names)},
                })
        return results

    def predict(self, image) -> dict:
        return self.predict_batch([image])[0]

    @torch.no_grad()
    def extract_features(self, images, normalize: bool = False) -> np.ndarray:
        single = not isinstance(images, (list, tuple))
        items = [images] if single else list(images)
        feats = [forward_features(self.model, x).cpu().numpy() for x in self._batches(items)]
        out = np.concatenate(feats, axis=0)
        if normalize:
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out[0] if single else out


def get_predictor(checkpoint=None, device="auto") -> Predictor:
    key = (str(checkpoint), str(device))
    if key not in _CACHE:
        _CACHE[key] = Predictor(checkpoint, device)
    return _CACHE[key]


def predict_image(image, checkpoint=None, device="auto") -> dict:
    return get_predictor(checkpoint, device).predict(image)


def predict_images(images, checkpoint=None, device="auto") -> list[dict]:
    return get_predictor(checkpoint, device).predict_batch(images)


def extract_features(images, checkpoint=None, device="auto", normalize: bool = False) -> np.ndarray:
    return get_predictor(checkpoint, device).extract_features(images, normalize=normalize)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Image paths or URLs")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint (default: newest best.pt in outputs/)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--features-out", default=None, help="Also save feature vectors to this .npy file")
    args = parser.parse_args()

    pred = get_predictor(args.checkpoint, args.device)
    print(f"# checkpoint: {pred.checkpoint}  ({pred.arch}, device: {pred.device})")
    for src, res in zip(args.images, pred.predict_batch(args.images)):
        print(json.dumps({"image": src, **res}, indent=2))
    if args.features_out:
        feats = pred.extract_features(args.images)
        np.save(args.features_out, feats)
        print(f"# saved features {feats.shape} to {args.features_out}")


if __name__ == "__main__":
    main()
