import torch
import torch.nn as nn


class DINOv2Extractor(nn.Module):
    """Wraps a torch.hub DINOv2 backbone. Frozen by default."""

    def __init__(self, variant: str = "dinov2_vits14", freeze: bool = True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", variant)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze = freeze

    def unfreeze_last_n_blocks(self, n: int):
        """Used in Task 9 (fine-tuning). Unfreezes the last n transformer blocks."""
        blocks = self.backbone.blocks
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x):
        # returns the CLS-token embedding, shape (batch, embedding_dim)
        return self.backbone(x)