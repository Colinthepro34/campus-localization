"""
Sweeps softmax_threshold values on the val set to find a threshold that
balances rejecting wrong predictions against rejecting correct ones.
Prints a small table; does not auto-write config.yaml (a human should
review the tradeoff and set the final value deliberately).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import yaml

from src.data.dataset import build_label_maps
from src.models.classifier_head import MLPClassifierHead


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    location_to_idx, _ = build_label_maps(cfg["data"]["metadata_csv"])
    num_classes = len(location_to_idx)

    data = np.load(f"{cfg['paths']['outputs_dir']}/embeddings/val_embeddings.npz", allow_pickle=True)
    X = torch.tensor(data["embeddings"], dtype=torch.float32).to(device)
    y_true = data["location_labels"]

    model = MLPClassifierHead(cfg["model"]["embedding_dim"], num_classes).to(device)
    model.load_state_dict(torch.load(f"{cfg['paths']['checkpoints_dir']}/dinov2_frozen_head_best.pt",
                                      map_location=device))
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1).cpu().numpy()

    print(f"{'threshold':>10} {'coverage':>10} {'accuracy_on_kept':>18}")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = probs.argmax(axis=1)
        top_probs = probs.max(axis=1)
        keep_mask = top_probs >= threshold
        coverage = keep_mask.mean()
        if keep_mask.sum() == 0:
            acc_on_kept = float("nan")
        else:
            acc_on_kept = (preds[keep_mask] == y_true[keep_mask]).mean()
        print(f"{threshold:>10.2f} {coverage:>10.2%} {acc_on_kept:>18.2%}")


if __name__ == "__main__":
    main()