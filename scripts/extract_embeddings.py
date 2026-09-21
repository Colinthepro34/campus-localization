"""
Precomputes and caches DINOv2 embeddings for every image in the dataset, so
that Task 6, Task 8, Task 11 and Task 12 don't each re-run the backbone.
Saves outputs/embeddings/{split}_embeddings.npz with arrays:
embeddings, location_labels, floor_labels, image_paths.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CampusLocationDataset, build_label_maps
from src.data.transforms import get_eval_transforms
from src.models.dinov2_extractor import DINOv2Extractor


def extract_split(model, loader, device):
    all_embeddings, all_loc, all_floor, all_paths = [], [], [], []
    with torch.no_grad():
        for images, loc_labels, floor_labels, paths in tqdm(loader, leave=False):
            images = images.to(device)
            emb = model(images).cpu().numpy()
            all_embeddings.append(emb)
            all_loc.append(loc_labels.numpy())
            all_floor.append(floor_labels.numpy())
            all_paths.extend(paths)
    return (
        np.concatenate(all_embeddings),
        np.concatenate(all_loc),
        np.concatenate(all_floor),
        np.array(all_paths),
    )


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = json.load(open(f"{cfg['data']['processed_dir']}/splits.json"))
    location_to_idx, floor_to_idx = build_label_maps(cfg["data"]["metadata_csv"])

    model = DINOv2Extractor(cfg["model"]["dinov2_variant"], freeze=True).to(device).eval()
    eval_tf = get_eval_transforms(cfg["data"]["image_size"])

    for split_name in ["train", "val", "test"]:
        ds = CampusLocationDataset(
            cfg["data"]["metadata_csv"], cfg["data"]["images_dir"], splits[split_name],
            location_to_idx, floor_to_idx, eval_tf,
        )
        loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"]["num_workers"])
        emb, loc, floor, paths = extract_split(model, loader, device)
        out_path = f"{cfg['paths']['outputs_dir']}/embeddings/{split_name}_embeddings.npz"
        np.savez(out_path, embeddings=emb, location_labels=loc, floor_labels=floor, image_paths=paths)
        print(f"{split_name}: saved {emb.shape} embeddings to {out_path}")


if __name__ == "__main__":
    main()