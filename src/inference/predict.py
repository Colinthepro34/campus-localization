"""
CLI entry point: given a path to a new photo, prints the predicted
location, floor, and confidence, e.g.:
  $ python -m src.inference.predict --image path/to/photo.jpg
  Prediction: Library -- Floor 1  (confidence: 0.87)
"""
import argparse

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.dataset import build_label_maps
from src.data.transforms import get_eval_transforms
from src.inference.confidence import classify_with_confidence
from src.models.classifier_head import MLPClassifierHead
from src.models.dinov2_extractor import DINOv2Extractor


def load_models(cfg, device):
    location_to_idx, floor_to_idx = build_label_maps(cfg["data"]["metadata_csv"])
    idx_to_location = {v: k for k, v in location_to_idx.items()}
    idx_to_floor = {v: k for k, v in floor_to_idx.items()}

    backbone = DINOv2Extractor(cfg["model"]["dinov2_variant"], freeze=True).to(device).eval()
    head = MLPClassifierHead(cfg["model"]["embedding_dim"], len(location_to_idx)).to(device)
    head.load_state_dict(torch.load(f"{cfg['paths']['checkpoints_dir']}/dinov2_frozen_head_best.pt",
                                     map_location=device))
    head.eval()
    return backbone, head, idx_to_location, idx_to_floor


def predict_image(image_path: str, cfg: dict, backbone, head, idx_to_location, device):
    transform = get_eval_transforms(cfg["data"]["image_size"])
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        emb = backbone(tensor)
        probs = torch.softmax(head(emb), dim=1).cpu().numpy()[0]

    pred_idx, confidence, is_confident = classify_with_confidence(
        probs, cfg["confidence"]["softmax_threshold"], cfg["confidence"]["margin_threshold"],
    )
    location = idx_to_location[pred_idx]
    return location, confidence, is_confident


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a new photo")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, head, idx_to_location, _idx_to_floor = load_models(cfg, device)

    location, confidence, is_confident = predict_image(args.image, cfg, backbone, head, idx_to_location, device)

    if is_confident:
        print(f"Prediction: {location}  (confidence: {confidence:.2f})")
    else:
        print(f"Uncertain -- best guess is {location} (confidence: {confidence:.2f}), "
              f"below the configured confidence threshold. Treating as unknown location.")


if __name__ == "__main__":
    main()