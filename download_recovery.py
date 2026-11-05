#!/usr/bin/env python
"""Download ReCOVery article images and create leakage-safe train/val/test manifests."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fakenews_cnn.config import ROOT

MAX_BYTES = 20 * 1024 * 1024
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ReCOVery-research-downloader/1.0)"}


def fetch_image(url: str, image_dir: Path, timeout: int, min_side: int, max_side: int):
    """Fetch and normalise one unique URL; return (url, relative_path, error)."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    out = image_dir / f"{digest}.jpg"
    if out.exists():
        return url, out.relative_to(ROOT).as_posix(), ""
    try:
        with requests.get(url, headers=HEADERS, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(64 * 1024):
                if chunk:
                    content.extend(chunk)
                    if len(content) > MAX_BYTES:
                        return url, "", "too_large"
        image = Image.open(io.BytesIO(content))
        image.load()
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = image.convert("RGB")
        if min(image.size) < min_side:
            return url, "", "too_small"
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        tmp = out.with_suffix(".jpg.tmp")
        image.save(tmp, format="JPEG", quality=90)
        tmp.replace(out)
        return url, out.relative_to(ROOT).as_posix(), ""
    except requests.RequestException as exc:
        return url, "", f"http:{type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 - malformed or unsupported image responses
        return url, "", f"image:{type(exc).__name__}"


def split_urls(df: pd.DataFrame, seed: int) -> dict[str, str]:
    """Stratify unique URLs 80/10/10, keeping duplicate images in one split."""
    group_labels = df.groupby("image", sort=True)["label"].agg(
        lambda labels: int(labels.mode().iloc[0])
    )
    urls = group_labels.index.to_numpy()
    labels = group_labels.to_numpy()
    train_urls, holdout_urls = train_test_split(
        urls, test_size=0.2, random_state=seed, stratify=labels
    )
    holdout_labels = group_labels.loc[holdout_urls].to_numpy()
    val_urls, test_urls = train_test_split(
        holdout_urls, test_size=0.5, random_state=seed, stratify=holdout_labels
    )
    return {
        **{url: "train" for url in train_urls},
        **{url: "val" for url in val_urls},
        **{url: "test" for url in test_urls},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/recovery-news-data.csv")
    parser.add_argument("--image-dir", default="data/images/recovery")
    parser.add_argument("--processed-dir", default="data/processed_recovery/subset")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--min-side", type=int, default=64)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = ROOT / image_dir
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, dtype={"news_id": str})
    source_rows = len(df)
    required = {"news_id", "title", "image", "reliability"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(required - set(df.columns))}")
    df["image"] = df["image"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["reliability"], errors="coerce")
    df = df[df["image"].str.match(r"^https?://", case=False) & df["label"].isin([0, 1])]
    df = df.drop_duplicates(subset="news_id").copy()
    df["label"] = df["label"].astype(int)
    if df.empty:
        raise RuntimeError("No rows with an HTTP image URL and binary reliability label.")
    valid_row_count = len(df)

    assignments = split_urls(df, args.seed)
    urls = list(assignments)
    results = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = [pool.submit(fetch_image, url, image_dir, args.timeout,
                            args.min_side, args.max_side) for url in urls]
        for future in tqdm(jobs, total=len(urls), desc="ReCOVery image URLs"):
            url, image_path, error = future.result()
            if error:
                failures.append({"image": url, "error": error})
            else:
                results[url] = image_path

    df = df[df["image"].isin(results)].copy()
    df["split"] = df["image"].map(assignments)
    df["image_path"] = df["image"].map(results)
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        manifest = pd.DataFrame({
            "id": part["news_id"].astype(str),
            "image_path": part["image_path"],
            "label": part["label"].astype(int),
            "title": part["title"].fillna("").astype(str),
        })
        if manifest.empty or manifest["label"].nunique() < 2:
            raise RuntimeError(f"Split '{split}' is empty or lacks one class after image downloads.")
        manifest.to_csv(processed_dir / f"{split}.csv", index=False)

    df.groupby("image", as_index=False).first()[["image", "split"]].to_csv(
        processed_dir / "image_split_assignments.csv", index=False
    )
    info = {
        "source_csv": str(csv_path),
        "source_rows": int(source_rows),
        "rows_with_valid_image_url_and_label": int(valid_row_count),
        "unique_image_urls": int(len(urls)),
        "successful_unique_images": int(len(results)),
        "failed_unique_images": int(len(failures)),
        "successful_article_rows": int(len(df)),
        "split_strategy": "stratified 80/10/10 by unique image URL; duplicate URLs share a split",
        "splits": {},
    }
    # `df` now contains only successfully downloaded records.
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        info["splits"][split] = {
            "samples": int(len(part)),
            "unique_images": int(part["image"].nunique()),
            "unreliable": int((part["label"] == 0).sum()),
            "reliable": int((part["label"] == 1).sum()),
        }
    with open(processed_dir / "failed_images.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    with open(processed_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
