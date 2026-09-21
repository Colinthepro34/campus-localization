import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Standard triplet margin loss on L2-normalized embeddings."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def sample_triplets(embeddings: torch.Tensor, labels: torch.Tensor, num_triplets: int):
    """Random triplet sampling: for each anchor, pick a random same-class
    positive and a random different-class negative."""
    anchors, positives, negatives = [], [], []
    labels_np = labels.cpu().numpy()
    for _ in range(num_triplets):
        anchor_idx = torch.randint(0, len(labels), (1,)).item()
        anchor_label = labels_np[anchor_idx]

        pos_candidates = (labels_np == anchor_label).nonzero()[0]
        pos_candidates = pos_candidates[pos_candidates != anchor_idx]
        if len(pos_candidates) == 0:
            continue
        pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]

        neg_candidates = (labels_np != anchor_label).nonzero()[0]
        neg_idx = neg_candidates[torch.randint(0, len(neg_candidates), (1,)).item()]

        anchors.append(embeddings[anchor_idx])
        positives.append(embeddings[pos_idx])
        negatives.append(embeddings[neg_idx])

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)