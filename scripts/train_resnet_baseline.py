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
from src.models.resnet_baseline import build_resnet50_classifier


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, loc_labels, _floor_labels, _paths in tqdm(loader, leave=False):
            images, loc_labels = images.to(device), loc_labels.to(device)
            outputs = model(images)
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

    model = build_resnet50_classifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr_baseline"],
                                   weight_decay=cfg["train"]["weight_decay"])

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg["train"]["epochs_baseline"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{cfg['paths']['checkpoints_dir']}/resnet50_baseline_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping.")
                break

    print(f"Best val acc: {best_val_acc:.4f}")
    with open(f"{cfg['paths']['outputs_dir']}/reports/resnet_baseline_summary.json", "w") as f:
        json.dump({"best_val_acc": best_val_acc}, f, indent=2)


if __name__ == "__main__":
    main()