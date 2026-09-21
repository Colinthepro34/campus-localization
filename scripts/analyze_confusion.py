"""
Loads the best DINOv2-frozen head + test embeddings, produces:
- outputs/figures/confusion_matrix.png
- outputs/reports/misclassified_images.csv (image_path, true_label, predicted_label, confidence)
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset import build_label_maps
from src.models.classifier_head import MLPClassifierHead


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    location_to_idx, _ = build_label_maps(cfg["data"]["metadata_csv"])
    idx_to_location = {v: k for k, v in location_to_idx.items()}
    num_classes = len(location_to_idx)

    data = np.load(f"{cfg['paths']['outputs_dir']}/embeddings/test_embeddings.npz", allow_pickle=True)
    X = torch.tensor(data["embeddings"], dtype=torch.float32).to(device)
    y_true = data["location_labels"]
    paths = data["image_paths"]

    model = MLPClassifierHead(cfg["model"]["embedding_dim"], num_classes).to(device)
    model.load_state_dict(torch.load(f"{cfg['paths']['checkpoints_dir']}/dinov2_frozen_head_best.pt",
                                      map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)

    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_names = [idx_to_location[i] for i in labels_present]

    report_text = classification_report(y_true, y_pred, labels=labels_present, target_names=label_names)
    print(report_text)
    with open(f"{cfg['paths']['outputs_dir']}/reports/classification_report.txt", "w") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    plt.figure(figsize=(max(6, len(labels_present) * 0.6), max(5, len(labels_present) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{cfg['paths']['outputs_dir']}/figures/confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {cfg['paths']['outputs_dir']}/figures/confusion_matrix.png")

    misclassified = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            misclassified.append({
                "image_path": paths[i],
                "true_label": idx_to_location[y_true[i]],
                "predicted_label": idx_to_location[y_pred[i]],
                "confidence": float(probs[i, y_pred[i]]),
            })
    mis_df = pd.DataFrame(misclassified).sort_values("confidence", ascending=False)
    mis_df.to_csv(f"{cfg['paths']['outputs_dir']}/reports/misclassified_images.csv", index=False)
    print(f"{len(mis_df)} misclassified test images written to misclassified_images.csv")


if __name__ == "__main__":
    main()