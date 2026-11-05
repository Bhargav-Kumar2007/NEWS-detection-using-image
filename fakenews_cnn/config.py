"""Config loading, path helpers and shared CLI arguments."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# Official Fakeddit 2_way_label: 0 = fake, 1 = real
CLASS_NAMES = ["fake", "real"]


def resolve(path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path="config.yaml") -> dict:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--full", action="store_true",
                        help="Use the FULL official dataset instead of the small subset")
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v2"], default=None,
                        help="Override model.arch from the config")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | mps (overrides config)")
    parser.add_argument("--num-workers", default=None, help="DataLoader workers (int or 'auto')")
    return parser


def apply_common_overrides(cfg: dict, args) -> dict:
    if getattr(args, "full", False):
        cfg["data"]["use_full_dataset"] = True
    if getattr(args, "model", None):
        cfg["model"]["arch"] = args.model
    if getattr(args, "device", None):
        cfg["device"] = args.device
    if getattr(args, "num_workers", None) is not None:
        v = args.num_workers
        cfg["data"]["num_workers"] = v if v == "auto" else int(v)
    return cfg


def dataset_mode(cfg: dict) -> str:
    return "full" if cfg["data"].get("use_full_dataset") else "subset"


def manifest_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["processed_dir"]) / dataset_mode(cfg)


def run_paths(cfg: dict, create: bool = False) -> dict:
    """Directories for the current run: outputs/<mode>_<arch>/{checkpoints,metrics,plots}."""
    base = resolve(cfg["paths"]["output_dir"]) / f"{dataset_mode(cfg)}_{cfg['model']['arch']}"
    paths = {
        "base": base,
        "checkpoints": base / "checkpoints",
        "metrics": base / "metrics",
        "plots": base / "plots",
    }
    if create:
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
    return paths


def positive_index(cfg: dict) -> int:
    name = str(cfg.get("eval", {}).get("positive_class", "fake")).lower()
    if name not in CLASS_NAMES:
        raise ValueError(f"eval.positive_class must be one of {CLASS_NAMES}, got '{name}'")
    return CLASS_NAMES.index(name)
