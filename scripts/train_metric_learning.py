"""
Trains a small projection head on top of frozen DINOv2 embeddings using
triplet loss, to sharpen the embedding space for retrieval / clustering.
This is a research extension, not the primary model.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.dataset import build_label_maps
from src.losses.triplet_loss import TripletLoss, sample_triplets


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    location_to_idx, _ = build_label_maps(cfg["data"]["metadata_csv"])

    train_data = np.load(f"{cfg['paths']['outputs_dir']}/embeddings/train_embeddings.npz", allow_pickle=True)
    X_train = torch.tensor(train_data["embeddings"], dtype=torch.float32).to(device)
    y_train = torch.tensor(train_data["location_labels"], dtype=torch.long).to(device)

    model = ProjectionHead(cfg["model"]["embedding_dim"]).to(device)
    criterion = TripletLoss(margin=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr_dinov2_head"])

    num_epochs = 50
    triplets_per_epoch = max(500, len(y_train) * 5)

    for epoch in range(num_epochs):
        model.train()
        anchors, positives, negatives = sample_triplets(X_train, y_train, triplets_per_epoch)
        proj_a, proj_p, proj_n = model(anchors), model(positives), model(negatives)
        loss = criterion(proj_a, proj_p, proj_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: triplet_loss={loss.item():.4f}")

    torch.save(model.state_dict(), f"{cfg['paths']['checkpoints_dir']}/metric_learning_projection.pt")
    with open(f"{cfg['paths']['outputs_dir']}/reports/metric_learning_summary.json", "w") as f:
        json.dump({"final_triplet_loss": loss.item()}, f, indent=2)
    print("Saved projection head. Use scripts/build_faiss_index.py to evaluate retrieval quality.")


if __name__ == "__main__":
    main()