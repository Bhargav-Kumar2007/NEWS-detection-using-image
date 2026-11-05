"""Datasets, transforms and DataLoaders built from the manifests written by download_data.py."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import ROOT, manifest_dir
from .utils import resolve_num_workers

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(round(image_size * 256 / 224))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FakedditImageDataset(Dataset):
    """Reads a manifest CSV with columns: id, image_path, label, title."""

    def __init__(self, manifest_csv, transform=None):
        self.df = pd.read_csv(manifest_csv, dtype={"id": str})
        self.transform = transform
        self.paths = [ROOT / p for p in self.df["image_path"]]
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def manifest_path(cfg: dict, split: str):
    return manifest_dir(cfg) / f"{split}.csv"


def make_loader(cfg: dict, split: str, train: bool, batch_size: int | None = None) -> DataLoader:
    size = int(cfg["data"]["image_size"])
    ds = FakedditImageDataset(manifest_path(cfg, split),
                              train_transform(size) if train else eval_transform(size))
    workers = resolve_num_workers(cfg["data"].get("num_workers", "auto"))
    bs = batch_size or (cfg["train"]["batch_size"] if train else cfg["eval"]["batch_size"])
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      drop_last=False)
