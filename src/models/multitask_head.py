import torch.nn as nn


class MultiTaskHead(nn.Module):
    """Shared trunk, two output heads: location and floor."""

    def __init__(self, embedding_dim: int, num_locations: int, num_floors: int,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.location_head = nn.Linear(hidden_dim, num_locations)
        self.floor_head = nn.Linear(hidden_dim, num_floors)

    def forward(self, x):
        shared = self.trunk(x)
        return self.location_head(shared), self.floor_head(shared)