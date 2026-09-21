"""
Same as train_dinov2_frozen.py, but re-extracts embeddings from augmented
images every epoch instead of loading cached ones. Compares final val_acc
against outputs/reports/dinov2_frozen_summary.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CampusLocationDataset, build_label_maps
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.classifier_head import MLPClassifierHead
from src.models.dinov2_extractor import DINOv2Extractor


def run_epoch(backbone, head, loader, criterion, optimizer, device, train: bool):
    head.train() if train else head.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, loc_labels, _floor_labels, _paths in tqdm(loader, leave=False):
            images, loc_labels = images.to(device), loc_labels.to(device)
            with torch.no_grad():
                emb = backbone(images)
            outputs = head(emb)
            loss = criterion(outputs, loc_labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == loc_labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = json.load(open(f"{cfg['data']['processed_dir']}/splits.json"))
    location_to_idx, floor_to_idx = build_label_maps(cfg["data"]["metadata_csv"])
    num_classes = len(location_to_idx)

    train_ds = CampusLocationDataset(
        cfg["data"]["metadata_csv"], cfg["data"]["images_dir"], splits["train"],
        location_to_idx, floor_to_idx, get_train_transforms(cfg["data"]["image_size"]),
    )
    val_ds = CampusLocationDataset(
        cfg["data"]["metadata_csv"], cfg["data"]["images_dir"], splits["val"],
        location_to_idx, floor_to_idx, get_eval_transforms(cfg["data"]["image_size"]),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"]["num_workers"])

    backbone = DINOv2Extractor(cfg["model"]["dinov2_variant"], freeze=True).to(device).eval()
    head = MLPClassifierHead(cfg["model"]["embedding_dim"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr_dinov2_head"],
                                   weight_decay=cfg["train"]["weight_decay"])

    best_val_acc = 0.0
    for epoch in range(cfg["train"]["epochs_dinov2_frozen"]):
        train_loss, train_acc = run_epoch(backbone, head, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(backbone, head, val_loader, criterion, optimizer, device, False)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(head.state_dict(), f"{cfg['paths']['checkpoints_dir']}/dinov2_augmented_head_best.pt")

    no_aug = json.load(open(f"{cfg['paths']['outputs_dir']}/reports/dinov2_frozen_summary.json"))
    comparison = {
        "no_augmentation_val_acc": no_aug["best_val_acc"],
        "with_augmentation_val_acc": best_val_acc,
        "delta": best_val_acc - no_aug["best_val_acc"],
    }
    print(comparison)
    with open(f"{cfg['paths']['outputs_dir']}/reports/augmentation_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()