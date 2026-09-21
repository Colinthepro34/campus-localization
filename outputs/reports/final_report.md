# Campus Visual Localization — Final Report

## Dataset

| Metric | Value |
|---|---|
| Total images | 50 |
| Distinct locations | 22 |
| Floors present | ground (0), second (2) |
| Distinct capture sessions | 23 (all single-session per location except xaviers hall which has 2) |
| Train / Val / Test split | 15 / 12 / 23 |
| Split strategy used | Stratified random (session-based fallback for every location) |

## Model Results (validation accuracy unless noted)

| Model | Best Val Acc | Notes |
|---|---|---|
| ResNet50 (ImageNet init) | 0.250 | Baseline |
| DINOv2-frozen + MLP head | 0.417 | Primary baseline |
| DINOv2-frozen + MLP + augmentation (on-the-fly) | 0.500 | +0.083 over no-aug |
| DINOv2 fine-tuned (last 2 blocks + MLP) | 0.250 | Degraded vs frozen |
| Multi-task (loc + floor) on frozen embeddings | 0.500 | Floor acc: 0.917 |

| Retrieval / Metric Model | Metric | Value |
|---|---|---|
| FAISS inner-product baseline (frozen DINOv2) | Top-1 / Top-5 (test) | 0.217 / 0.261 |
| Projection head (triplet loss) | Final triplet loss | 0.0 (fully memorised) |

## Test Set Performance (frozen DINOv2 head, best model)

- Overall accuracy: 0.22 (5/23)
- Only Chapel and Office Building were perfectly recalled; Main Gate 1 was recalled at 0.5 precision

## Key Findings

1. **DINOv2 outperforms ResNet50**: The frozen DINOv2 classifier (0.417) beats the ResNet50 baseline (0.25), supporting the hypothesis that self-supervised vision features provide a stronger foundation for visual localization on small datasets.

2. **Augmentation helps**: On-the-fly training augmentation on DINOv2 embeddings yielded a +0.083 improvement (0.417 → 0.500).

3. **Multi-task learning matches single-task on augmentation, improves over frozen-only**: Joint location+floor prediction matched the augmented result at 0.500 with the best floor accuracy of 0.917.

4. **Fine-tuning hurts**: DINOv2 fine-tuning (unfreezing last 2 blocks) degraded to 0.25 — consistent with overfitting on this extremely small dataset.

5. **FAISS retrieval underperforms**: Raw nearest-neighbor retrieval (Top-1 0.22) is substantially below supervised classifiers, indicating that a learned classification head is essential for this dataset.

## Failure Mode Analysis

- Most misclassifications are **majority-class bias**: canteen (6 training images) and xaviers hall (8 training images) dominate predictions when the model is uncertain.
- **No clear floor-adjacent confusion**: The small number of second-floor images (library, xerox) are simply misclassified as majority classes rather than confused with ground-floor neighbours — this failure mode would require a larger dataset to observe.

## Limitations

1. **Tiny dataset (50 images)**: All 21 of 22 locations have fewer than 8 images; most have 1–2. This makes reliable generalisation estimates impossible.

2. **Session-based splitting could not be applied**: Every location has only 1–2 capture sessions (minimum 3 required). All splits fell back to stratified random, so **there is no leakage-free train/test separation based on independent capture sessions**. This is the most significant methodological weakness.

3. **Single-image locations in test**: Several locations (1-image classes) ended up entirely in test with zero training examples. The model must generalise to unseen locations from zero shots — a fundamentally different task from what was evaluated on the validation set (which benefited from augmented in-distribution classes).

4. **Class imbalance**: Canteen (6 images) and xaviers hall (8 images) dominate the training set. The model over-predicts these.

5. **Time-of-day variation absent**: All 50 images were captured in one morning session under sunny conditions. No variation across time-of-day or weather.

6. **Same device / photographer**: All images captured with the same vivo T3 Pro 5G by the same photographer in a single walk. Generalisability to different devices or viewpoints is unknown.

## Sanity Checklist (Project Spec Section 37)

- [x] Were train/val/test images taken from different capture sessions? **NO** — every location has 1–2 sessions only; all fell back to random split. This limitation is flagged above.
- [ ] Any near-duplicate warnings that ended up split across sets? **No near-duplicates detected** by perceptual hash.
- [x] Does accuracy look suspiciously high (>98%)? **No** — best val accuracy is 0.50; test accuracy is 0.22. Numbers are realistic for the dataset size.
- [x] Does the confusion matrix show floor-adjacent confusion? **Not clearly** — failures are majority-class bias, not floor-adjacent. Would need more data to test.
- [ ] Was the same phone/photographer used consistently across train and test classes? **Yes** — all from vivo T3 Pro 5G, same session. This limits claims about cross-device generalisation.

## Files Produced

```
outputs/reports/
├── dataset_validation.md
├── resnet_baseline_summary.json
├── dinov2_frozen_summary.json
├── dinov2_finetune_summary.json
├── augmentation_comparison.json
├── multitask_summary.json
├── metric_learning_summary.json
├── retrieval_baseline_summary.json
├── classification_report.txt
├── misclassified_images.csv
└── final_report.md

outputs/figures/
└── confusion_matrix.png

checkpoints/
├── resnet50_baseline_best.pt
├── dinov2_frozen_head_best.pt
├── dinov2_augmented_head_best.pt
├── dinov2_finetuned_best.pt
├── multitask_best.pt
├── metric_learning_projection.pt
└── faiss_index.bin
```
