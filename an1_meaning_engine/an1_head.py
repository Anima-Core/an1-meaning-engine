"""AN1 tiny MLP head for 64D header -> logits."""

import torch.nn as nn


class AN1Head(nn.Module):
    """Tiny MLP that maps 64D intention header to 10 class logits."""

    def __init__(self, in_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

