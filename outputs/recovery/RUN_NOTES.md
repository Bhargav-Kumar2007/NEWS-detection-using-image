# ReCOVery CNN run

Run date: 2026-11-05  
Dataset: [ReCOVery](https://github.com/apurvamulay/ReCOVery), official `recovery-news-data.csv`  
Task: image-only reliable vs. unreliable classification; label 0 = unreliable/fake, label 1 = reliable/real.

## Data collection and splits

- Source CSV rows: 2,029
- Rows with a valid image URL and binary label: 2,017
- Unique image URLs attempted: 1,939
- Images downloaded and normalized: 1,407
- Failed image URLs: 532 (details in `data/processed_recovery/subset/failed_images.json`)
- Successful article rows: 1,436
- Split method: stratified 80/10/10 by unique image URL, keeping duplicate URLs in a single split.

| Split | Rows | Unique images | Unreliable/fake | Reliable/real |
|---|---:|---:|---:|---:|
| Train | 1,157 | 1,129 | 134 | 1,023 |
| Validation | 137 | 137 | 15 | 122 |
| Test | 142 | 141 | 19 | 123 |

## Training

- Model: ImageNet-pretrained ResNet18
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA, mixed precision)
- Epochs completed: 20
- Best checkpoint: epoch 10, selected by validation fake-class F1 = 0.4211
- Checkpoint: `outputs/recovery/subset_resnet18/checkpoints/best.pt`

## Final held-out results

The best checkpoint was evaluated at the standard 0.5 probability threshold, with `fake` as the positive class.

| Split | Accuracy | Fake precision | Fake recall | Fake F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation (n=137) | 0.9197 | 1.0000 | 0.2667 | 0.4211 | 0.6727 |
| Test (n=142) | 0.9014 | 1.0000 | 0.2632 | 0.4167 | 0.7617 |

Test confusion matrix (rows = true class, columns = predicted class; order fake, real):

```text
[[  5, 14],
 [  0,123]]
```

The model is conservative when predicting fake: it found 5 of 19 fake test examples and mislabeled no real examples as fake. Despite 90.1% overall test accuracy, fake recall is low and the small validation/test fake counts make the class-specific estimates noisy. Treat this checkpoint as a baseline, not a reliable fact-checking model.

## Artifacts

- Full metrics, classification reports, and prediction CSVs: `outputs/recovery/subset_resnet18/metrics/`
- Epoch history and training summary: `outputs/recovery/subset_resnet18/metrics/history.csv`, `training_summary.json`
- Plots: `outputs/recovery/subset_resnet18/plots/`
- Dataset split and download records: `data/processed_recovery/subset/`
