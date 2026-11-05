# Fakeddit CNN - Fake News Image Classification (2-way)

A plug-and-play PyTorch project that classifies [Fakeddit](https://github.com/entitize/Fakeddit)
post **images** as `fake` (label 0) or `real` (label 1) using an ImageNet-pretrained
**ResNet18** or **MobileNetV2**.

* Uses the official `2_way_label` and the official train / validation / test_public splits.
* Downloads and preprocesses the TSV files and the images automatically.
* Small default subset for a quick test run, `--full` for the complete dataset.
* Automatic CPU / CUDA (/ Apple MPS) detection.
* Saves checkpoints, metrics (Accuracy, Precision, Recall, F1, ROC-AUC), classification report,
  confusion matrix, ROC curve and training curves automatically.
* Reusable `predict_image` and `extract_features` functions.

> No results are included in this repository. Every number and plot is produced by *your* run.

## Project layout

```
fakeddit_cnn/
├── train.py            # training + validation (+ test evaluation at the end)
├── evaluate.py         # metrics, report, confusion matrix, ROC curve
├── download_data.py    # TSV download, image download, preprocessing, manifests
├── predict.py          # predict_image / predict_images / extract_features / Predictor
├── config.yaml         # all settings
├── requirements.txt
├── run.sh / run.bat    # Linux-macOS / Windows runners
├── .gitignore
└── fakenews_cnn/       # shared modules (models, data, engine, metrics, plotting, utils)
```

Created automatically when you run things: `data/` (raw TSVs, images, manifests) and
`outputs/<subset|full>_<arch>/` (checkpoints, metrics, plots, logs).

## Quick start

Requires Python 3.9+ and an internet connection.

### Linux / macOS
```bash
chmod +x run.sh
./run.sh all            # setup + download small subset + train + evaluate
```

### Windows (Command Prompt or PowerShell)
```bat
run.bat all
```
`run.bat setup` installs the CUDA build of PyTorch automatically if an NVIDIA GPU is found
(via `nvidia-smi`); otherwise the CPU build.

### Google Colab
Upload/unzip the project, choose a GPU runtime (Runtime > Change runtime type), then:
```python
%cd fakeddit_cnn
!pip install -q -r requirements.txt
!python train.py                 # downloads data automatically, trains, evaluates
```
For the full dataset: `!python train.py --full`. To keep results, copy `outputs/` to Google Drive.

### Individual steps
```bash
./run.sh download                 # (or: python download_data.py)
./run.sh train                    # (or: python train.py)
./run.sh evaluate --split test    # (or: python evaluate.py --split test)
./run.sh predict my_image.jpg     # (or: python predict.py my_image.jpg)
```
(`run.bat` takes the same commands on Windows.)

## Subset vs. full dataset

| Mode   | Command                    | Size (from `config.yaml`)                      |
|--------|----------------------------|------------------------------------------------|
| Subset | `python train.py`          | `data.subset`: 2000 train / 400 val / 400 test |
| Full   | `python train.py --full`   | every official sample whose image can be fetched |

Edit `data.subset` in `config.yaml` to change subset sizes. Subset and full runs keep separate
manifests and outputs, so they never overwrite each other. Images are cached in `data/images/`
and shared between modes. **Full mode** means several hundred thousand image downloads: expect many
hours and a lot of disk/bandwidth; the download resumes where it stopped if interrupted.

## Models and settings

Set in `config.yaml` or on the command line:
```bash
python train.py --model mobilenet_v2 --epochs 15 --batch-size 64 --lr 0.0005
python train.py --device cpu
python train.py --resume          # continue from outputs/<run>/checkpoints/last.pt
```
Notable options: `model.freeze_backbone` (train head only; faster on CPU), `train.class_weights`
(inverse-frequency loss weights), `train.monitor` (criterion for the best checkpoint),
`train.early_stopping_patience`, `eval.positive_class`.

## Outputs

`outputs/<subset|full>_<arch>/`
```
checkpoints/best.pt, last.pt
metrics/history.json, history.csv, training_summary.json,
        metrics_{val,test}.json, classification_report_{val,test}.txt, predictions_{val,test}.csv
plots/  training_curves.png, accuracy_curve.png, loss_curve.png,
        confusion_matrix_{val,test}.png, roc_curve_{val,test}.png
train.log, config_used.json
```
`metrics_*.json` contains Accuracy, Precision, Recall, F1, macro-F1 and ROC-AUC.
Precision / Recall / F1 / ROC-AUC refer to the positive class set by `eval.positive_class`
(default `fake`). The classification report and confusion matrix cover both classes.

## Using the model in your own code

```python
from predict import predict_image, predict_images, extract_features, Predictor

predict_image("photo.jpg")
# {'label': 'fake', 'label_id': 0, 'confidence': ..., 'probabilities': {'fake': ..., 'real': ...}}

predict_images(["a.jpg", "https://example.com/b.png"])
feats = extract_features(["a.jpg", "b.jpg"])      # (N, 512) for ResNet18, (N, 1280) for MobileNetV2

p = Predictor("outputs/subset_resnet18/checkpoints/best.pt")   # explicit checkpoint, reusable
p.predict("photo.jpg"); p.extract_features("photo.jpg", normalize=True)
```
Without a `checkpoint` argument, the newest `outputs/*/checkpoints/best.pt` is used.

## Data notes and caveats

* **Labels:** `2_way_label` 0 = fake, 1 = real. Class balance depends on the split; check
  `data/processed/<mode>/dataset_info.json` after download. Prefer F1 / ROC-AUC over accuracy alone.
* **Dead links:** Many Reddit/Imgur image URLs in Fakeddit no longer work. The downloader skips
  them (and small placeholder images), and keeps requesting until the target subset size is reached
  or candidates run out. Failures are logged in `data/images/failed_ids.txt`
  (`--retry-failed` retries them).
* **Google Drive:** The TSV files come from the Drive folder linked in the Fakeddit README
  (`data.gdrive_folder_url`). Drive can rate-limit automated downloads. If it fails, the script
  prints what to do: download the files in your browser, drop them anywhere inside `data/raw/`
  (or pass `--tsv-dir`), and re-run. Needed files: `multimodal_train.tsv`, `multimodal_validate.tsv`,
  `multimodal_test_public.tsv` (or the `all_*` equivalents with `data.source: all`). If the Drive
  link changes, update it in `config.yaml`.
* **Image only:** this model sees only the image, not the title text. Fakeddit labels come from
  subreddit-level distant supervision, so the classifier learns dataset patterns; it is not a
  fact-checker and its predictions should not be used as proof of truth or falsehood.
* **Windows:** DataLoader workers default to 0 (`data.num_workers: auto`) for reliability.
* Pretrained ImageNet weights are downloaded by torchvision on first use (needs internet once).

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | `--batch-size 16` |
| Training very slow on CPU | `model.freeze_backbone: true`, `--model mobilenet_v2`, smaller subset |
| `Could not obtain these Fakeddit files` | Follow the manual TSV steps above |
| Fewer images than requested | Dead links; lower `data.subset` or accept the smaller set |
| Want a clean restart | Delete `outputs/<run>/` or run `python download_data.py --force` |

## Citation

Nakamura, K., Levy, S., Wang, W. Y. *r/Fakeddit: A New Multimodal Benchmark Dataset for
Fine-grained Fake News Detection.* LREC 2020. Please also follow the dataset's own terms of use.
