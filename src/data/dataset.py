from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CampusLocationDataset(Dataset):
    """
    Reads image_path/location/floor rows (filtered to a given split) and
    returns (image_tensor, location_label_idx, floor_label_idx, image_path).
    """

    def __init__(self, metadata_csv: str, images_dir: str, split_paths: list,
                 location_to_idx: dict, floor_to_idx: dict, transforms=None):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["image_path"].isin(split_paths)].reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.location_to_idx = location_to_idx
        self.floor_to_idx = floor_to_idx
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["image_path"]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        location_idx = self.location_to_idx[row["location"]]
        floor_idx = self.floor_to_idx[str(row["floor"])]
        return image, location_idx, floor_idx, row["image_path"]


def build_label_maps(metadata_csv: str):
    df = pd.read_csv(metadata_csv)
    locations = sorted(df["location"].unique().tolist())
    floors = sorted(df["floor"].astype(str).unique().tolist())
    location_to_idx = {loc: i for i, loc in enumerate(locations)}
    floor_to_idx = {fl: i for i, fl in enumerate(floors)}
    return location_to_idx, floor_to_idx