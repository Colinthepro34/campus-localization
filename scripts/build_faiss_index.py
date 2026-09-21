"""
Builds a FAISS index from train embeddings and evaluates nearest-neighbor
retrieval accuracy on test embeddings. This is a comparison baseline per
the project spec, not the primary system.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import numpy as np
import yaml

from src.data.dataset import build_label_maps


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    location_to_idx, _ = build_label_maps(cfg["data"]["metadata_csv"])
    idx_to_location = {v: k for k, v in location_to_idx.items()}

    train_data = np.load(f"{cfg['paths']['outputs_dir']}/embeddings/train_embeddings.npz", allow_pickle=True)
    test_data = np.load(f"{cfg['paths']['outputs_dir']}/embeddings/test_embeddings.npz", allow_pickle=True)

    train_emb = train_data["embeddings"].astype("float32")
    train_labels = train_data["location_labels"]
    test_emb = test_data["embeddings"].astype("float32")
    test_labels = test_data["location_labels"]

    faiss.normalize_L2(train_emb)
    faiss.normalize_L2(test_emb)

    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)

    k = 5
    _distances, indices = index.search(test_emb, k)

    top1_correct, top5_correct = 0, 0
    for i in range(len(test_labels)):
        neighbor_labels = train_labels[indices[i]]
        if neighbor_labels[0] == test_labels[i]:
            top1_correct += 1
        if test_labels[i] in neighbor_labels:
            top5_correct += 1

    top1_acc = top1_correct / len(test_labels)
    top5_acc = top5_correct / len(test_labels)
    print(f"Retrieval Top-1: {top1_acc:.4f}  Top-5: {top5_acc:.4f}")

    faiss.write_index(index, f"{cfg['paths']['checkpoints_dir']}/faiss_index.bin")
    with open(f"{cfg['paths']['outputs_dir']}/reports/retrieval_baseline_summary.json", "w") as f:
        json.dump({"top1_acc": top1_acc, "top5_acc": top5_acc}, f, indent=2)


if __name__ == "__main__":
    main()