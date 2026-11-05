#!/usr/bin/env bash
# Linux / macOS runner.   Usage:  ./run.sh <command> [options]
#   setup      create .venv and install requirements
#   download   download + preprocess data      (add --full for the full dataset)
#   train      train (auto-downloads data)     (add --full, --model, --epochs, ...)
#   evaluate   evaluate best checkpoint        (--split val|test, --checkpoint ...)
#   predict    predict images                  (./run.sh predict img1.jpg img2.png)
#   all        setup (if needed) + download + train + evaluate
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY_BIN="${PYTHON:-python3}"

setup() {
  if [ ! -d .venv ]; then "$PY_BIN" -m venv .venv; fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo "Setup complete."
}

activate() {
  if [ ! -d .venv ]; then setup; fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  setup)    setup ;;
  download) activate; python download_data.py "$@" ;;
  train)    activate; python train.py "$@" ;;
  evaluate) activate; python evaluate.py "$@" ;;
  predict)  activate; python predict.py "$@" ;;
  all)
    activate
    full=""
    for a in "$@"; do if [ "$a" = "--full" ]; then full="--full"; fi; done
    python download_data.py $full
    python train.py "$@"
    ;;
  *) sed -n '2,9p' "$0" ;;
esac
