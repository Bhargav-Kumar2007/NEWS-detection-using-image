#!/usr/bin/env python
"""Download + preprocess Fakeddit (official 2-way labels, official train/val/test splits).

Steps
  1. Fetch the official Fakeddit .tsv files (Google Drive via gdown) into data/raw/
     (skipped when the files are already there).
  2. Read the official splits: train / validate / test_public, keep rows with a valid image URL
     and a valid 2_way_label (0 = fake, 1 = real).
  3. Download the images in parallel, validate them, convert to RGB JPEG and resize
     (data/images/{id}.jpg). Dead links are skipped and remembered in data/images/failed_ids.txt.
  4. Write manifests data/processed/<subset|full>/{train,val,test}.csv.

Usage
  python download_data.py            # small subset (sizes from config.yaml)
  python download_data.py --full     # full official dataset (hundreds of thousands of images!)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from fakenews_cnn.config import (ROOT, add_common_args, apply_common_overrides, dataset_mode,
                                 load_config, manifest_dir, resolve)
from fakenews_cnn.utils import get_logger, save_json

SPLIT_FILES = {
    "multimodal": {"train": "multimodal_train.tsv", "val": "multimodal_validate.tsv",
                   "test": "multimodal_test_public.tsv"},
    "all": {"train": "all_train.tsv", "val": "all_validate.tsv", "test": "all_test_public.tsv"},
}
SPLITS = ("train", "val", "test")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fakeddit-research-downloader/1.0)"}
MAX_BYTES = 20 * 1024 * 1024
CHUNK_LIMIT = 20000
log = get_logger("download")


# --------------------------------------------------------------------------- TSV acquisition
def _extract_archives(root: Path) -> None:
    for z in list(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
            log.info(f"Extracted {z.name}")
        except zipfile.BadZipFile:
            log.warning(f"Could not extract {z}")
    for g in list(root.rglob("*.tsv.gz")):
        target = g.with_suffix("")
        if not target.exists():
            with gzip.open(g, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            log.info(f"Extracted {g.name}")


def _find_tsv(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == filename.lower():
            return p
    return None


def ensure_tsvs(cfg: dict, tsv_dir: str | None = None, skip_download: bool = False) -> dict:
    source = cfg["data"]["source"]
    wanted = SPLIT_FILES[source]
    root = Path(tsv_dir).expanduser().resolve() if tsv_dir else resolve(cfg["paths"]["raw_dir"])
    root.mkdir(parents=True, exist_ok=True)

    def locate():
        return {s: _find_tsv(root, wanted[s]) for s in SPLITS}

    _extract_archives(root)
    found = locate()
    if all(found.values()):
        log.info(f"Found Fakeddit TSV files in {root}")
        return found

    if not skip_download and not tsv_dir:
        url = cfg["data"]["gdrive_folder_url"]
        log.info("Downloading Fakeddit TSV files from Google Drive (this can take a few minutes)...")
        try:
            import gdown
            # `remaining_ok` was removed in newer gdown releases.
            gdown.download_folder(url=url, output=str(root), quiet=False)
        except Exception as e:  # noqa: BLE001
            log.error(f"Automatic download failed: {e}")
        _extract_archives(root)
        found = locate()

    missing = [wanted[s] for s in SPLITS if not found[s]]
    if missing:
        raise FileNotFoundError(
            "Could not obtain these Fakeddit files: " + ", ".join(missing) + "\n"
            "Google Drive sometimes blocks automated downloads (quota / rate limits).\n"
            "Fix: open the data link from https://github.com/entitize/Fakeddit in your browser,\n"
            f"download the files above and put them anywhere inside:\n    {root}\n"
            "then run this script again (it will not re-download what it already has)."
        )
    return found


# --------------------------------------------------------------------------- table loading
def load_split_table(tsv: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip",
                     dtype=str, encoding="utf-8", encoding_errors="replace")
    need = {"id", "image_url", label_col}
    if not need.issubset(df.columns):
        raise ValueError(f"{tsv.name} is missing columns {need - set(df.columns)}; "
                         f"found {list(df.columns)}")
    n0 = len(df)
    if "hasImage" in df.columns:
        df = df[df["hasImage"].astype(str).str.lower() == "true"]
    df = df[df["image_url"].fillna("").str.startswith(("http://", "https://"))]
    df = df[df[label_col].isin(["0", "1"])]
    df = df.drop_duplicates(subset="id")
    title_col = "clean_title" if "clean_title" in df.columns else ("title" if "title" in df.columns else None)
    out = pd.DataFrame({
        "id": df["id"].astype(str),
        "image_url": df["image_url"],
        "label": df[label_col].astype(int),
        "title": df[title_col].fillna("") if title_col else "",
    }).reset_index(drop=True)
    log.info(f"{tsv.name}: {n0} rows -> {len(out)} usable rows with image URL and valid label")
    return out


# --------------------------------------------------------------------------- image download
def _safe_name(post_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", post_id)


def _fetch_image(url: str, out_path: Path, timeout: int, max_side: int, min_side: int):
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return False, f"http_{r.status_code}"
        if len(r.content) > MAX_BYTES:
            return False, "too_large"
        img = Image.open(io.BytesIO(r.content))
        img.load()
        if getattr(img, "is_animated", False):
            img.seek(0)
        img = img.convert("RGB")
        w, h = img.size
        if min(w, h) < min_side:
            return False, "too_small"
        if max(w, h) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        tmp = out_path.with_name(out_path.name + ".tmp")
        img.save(tmp, format="JPEG", quality=90)
        tmp.replace(out_path)
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__


def _load_failed(path: Path) -> dict:
    failed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                failed[k] = v
    return failed


def collect_images(df: pd.DataFrame, target: int | None, cfg: dict, failed: dict,
                   failed_file: Path, desc: str, retry_failed: bool = False) -> pd.DataFrame:
    d = cfg["data"]
    images_dir = resolve(cfg["paths"]["images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    target = n if target is None else min(int(target), n)
    kept, pos, attempted = [], 0, 0
    stats: dict[str, int] = {}
    pbar = tqdm(total=target, desc=desc)

    def work(row):
        out = images_dir / f"{_safe_name(row.id)}.jpg"
        if out.exists():
            return row, True, "cached"
        if not retry_failed and row.id in failed:
            return row, False, "known_failed"
        ok, reason = _fetch_image(row.image_url, out, d["download_timeout"],
                                  d["stored_max_side"], d["min_side"])
        return row, ok, reason

    with ThreadPoolExecutor(max_workers=int(d["download_workers"])) as ex:
        while len(kept) < target and pos < n:
            need = target - len(kept)
            step = min(n - pos, CHUNK_LIMIT, max(int(need * 1.5) + 16, 64))
            batch = list(df.iloc[pos:pos + step].itertuples(index=False))
            pos += step
            new_fail = []
            for row, ok, reason in ex.map(work, batch):
                attempted += 1
                stats[reason] = stats.get(reason, 0) + 1
                if ok:
                    if len(kept) < target:
                        kept.append(row)
                        pbar.update(1)
                elif reason != "known_failed":
                    new_fail.append((row.id, reason))
            if new_fail:
                failed.update(dict(new_fail))
                with open(failed_file, "a", encoding="utf-8") as f:
                    f.writelines(f"{k}\t{v}\n" for k, v in new_fail)
    pbar.close()
    log.info(f"{desc}: kept {len(kept)}/{target} requested images "
             f"(tried {attempted}; outcomes: {stats})")
    if len(kept) < target:
        log.warning(f"{desc}: only {len(kept)} images available (many Reddit/Imgur links are dead).")

    rows = [{
        "id": r.id,
        "image_path": (images_dir / f"{_safe_name(r.id)}.jpg").relative_to(ROOT).as_posix()
        if images_dir.is_relative_to(ROOT) else str(images_dir / f"{_safe_name(r.id)}.jpg"),
        "label": r.label,
        "title": r.title,
    } for r in kept]
    return pd.DataFrame(rows, columns=["id", "image_path", "label", "title"])


# --------------------------------------------------------------------------- orchestration
def manifests_exist(cfg: dict) -> bool:
    mdir = manifest_dir(cfg)
    return all((mdir / f"{s}.csv").exists() for s in SPLITS)


def prepare_data(cfg: dict, tsv_dir: str | None = None, skip_tsv_download: bool = False,
                 force: bool = False, retry_failed: bool = False) -> None:
    if manifests_exist(cfg) and not force:
        log.info(f"Manifests for '{dataset_mode(cfg)}' mode already exist in {manifest_dir(cfg)} "
                 f"(use --force to rebuild).")
        return

    mode = dataset_mode(cfg)
    log.info(f"Preparing Fakeddit data in '{mode}' mode")
    if mode == "full":
        log.warning("FULL mode downloads several hundred thousand images (tens of GB of traffic, "
                    "hours of runtime). Re-running resumes from cached images.")

    files = ensure_tsvs(cfg, tsv_dir=tsv_dir, skip_download=skip_tsv_download)
    label_col = cfg["data"]["label_column"]
    seed = int(cfg["seed"])
    failed_file = resolve(cfg["paths"]["images_dir"]) / "failed_ids.txt"
    failed_file.parent.mkdir(parents=True, exist_ok=True)
    failed = {} if retry_failed else _load_failed(failed_file)

    mdir = manifest_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    info = {"mode": mode, "source": cfg["data"]["source"], "label_column": label_col,
            "label_mapping": {"0": "fake", "1": "real"}, "splits": {}}

    for split in SPLITS:
        table = load_split_table(files[split], label_col)
        table = table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        target = None if mode == "full" else cfg["data"]["subset"][split]
        manifest = collect_images(table, target, cfg, failed, failed_file,
                                  desc=f"{split} images", retry_failed=retry_failed)
        if manifest.empty:
            raise RuntimeError(f"No images could be downloaded for split '{split}'. "
                               "Check your internet connection / proxy settings.")
        manifest.to_csv(mdir / f"{split}.csv", index=False)
        counts = manifest["label"].value_counts().to_dict()
        info["splits"][split] = {"n_samples": int(len(manifest)),
                                 "n_fake": int(counts.get(0, 0)), "n_real": int(counts.get(1, 0)),
                                 "source_tsv": files[split].name}
        log.info(f"{split}: {len(manifest)} samples (fake={counts.get(0, 0)}, real={counts.get(1, 0)})")

    save_json(info, mdir / "dataset_info.json")
    log.info(f"Done. Manifests written to {mdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--tsv-dir", default=None,
                        help="Folder that already contains the Fakeddit .tsv files (skips Drive download)")
    parser.add_argument("--skip-tsv-download", action="store_true",
                        help="Never try Google Drive; only use files already in data/raw")
    parser.add_argument("--force", action="store_true", help="Rebuild manifests even if they exist")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry image URLs that failed in earlier runs")
    args = parser.parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    prepare_data(cfg, tsv_dir=args.tsv_dir, skip_tsv_download=args.skip_tsv_download,
                 force=args.force, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
