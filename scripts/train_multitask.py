"""
Trains MultiTaskHead on cached DINOv2 embeddings, jointly predicting
location and floor. Total loss = location_loss + floor_loss_weight * floor_loss.
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

from src.data.dataset import build_label_maps
from src.models.multitask_head import MultiTaskHead

FLOOR_LOSS_WEIGHT = 0.5


def load_split(cfg, split_name):
    data = np.load(f"{cfg['paths']['outputs_dir']}/embeddings/{split_name}_embeddings.npz",
                    allow_pickle=True)
    X = torch.tensor(data["embeddings"], dtype=torch.float32)
    y_loc = torch.tensor(data["location_labels"], dtype=torch.long)
    y_floor = torch.tensor(data["floor_labels"], dtype=torch.long)
    return TensorDataset(X, y_loc, y_floor)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, loc_correct, floor_correct, total = 0.0, 0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y_loc, y_floor in loader:
            X, y_loc, y_floor = X.to(device), y_loc.to(device), y_floor.to(device)
            loc_logits, floor_logits = model(X)
            loss = criterion(loc_logits, y_loc) + FLOOR_LOSS_WEIGHT * criterion(floor_logits, y_floor)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            loc_correct += (loc_logits.argmax(1) == y_loc).sum().item()
            floor_correct += (floor_logits.argmax(1) == y_floor).sum().item()
            total += X.size(0)
    return total_loss / total, loc_correct / total, floor_correct / total


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    location_to_idx, floor_to_idx = build_label_maps(cfg["data"]["metadata_csv"])

    train_ds = load_split(cfg, "train")
    val_ds = load_split(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = MultiTaskHead(cfg["model"]["embedding_dim"], len(location_to_idx), len(floor_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr_dinov2_head"],
                                   weight_decay=cfg["train"]["weight_decay"])

    best_val_loc_acc = 0.0
    for epoch in range(cfg["train"]["epochs_dinov2_frozen"]):
        train_loss, train_loc_acc, train_floor_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_loc_acc, val_floor_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"Epoch {epoch+1}: val_loc_acc={val_loc_acc:.4f} val_floor_acc={val_floor_acc:.4f}")
        if val_loc_acc > best_val_loc_acc:
            best_val_loc_acc = val_loc_acc
            torch.save(model.state_dict(), f"{cfg['paths']['checkpoints_dir']}/multitask_best.pt")

    print(f"Best val location acc: {best_val_loc_acc:.4f}")
    with open(f"{cfg['paths']['outputs_dir']}/reports/multitask_summary.json", "w") as f:
        json.dump({"best_val_location_acc": best_val_loc_acc}, f, indent=2)


if __name__ == "__main__":
    main()