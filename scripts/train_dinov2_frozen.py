"""
Trains an MLP classifier head on top of the precomputed frozen DINOv2
embeddings from Task 6's extract_embeddings.py. This is the project's
primary/main baseline.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.dataset import build_label_maps
from src.models.classifier_head import MLPClassifierHead


def load_split(cfg, split_name):
    data = np.load(f"{cfg['paths']['outputs_dir']}/embeddings/{split_name}_embeddings.npz",
                    allow_pickle=True)
    X = torch.tensor(data["embeddings"], dtype=torch.float32)
    y = torch.tensor(data["location_labels"], dtype=torch.long)
    return TensorDataset(X, y)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    location_to_idx, _ = build_label_maps(cfg["data"]["metadata_csv"])
    num_classes = len(location_to_idx)

    train_ds = load_split(cfg, "train")
    val_ds = load_split(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = MLPClassifierHead(cfg["model"]["embedding_dim"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr_dinov2_head"],
                                   weight_decay=cfg["train"]["weight_decay"])

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg["train"]["epochs_dinov2_frozen"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{cfg['paths']['checkpoints_dir']}/dinov2_frozen_head_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping.")
                break

    print(f"Best val acc: {best_val_acc:.4f}")
    with open(f"{cfg['paths']['outputs_dir']}/reports/dinov2_frozen_summary.json", "w") as f:
        json.dump({"best_val_acc": best_val_acc}, f, indent=2)


if __name__ == "__main__":
    main()