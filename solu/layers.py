import torch
import torch.nn as nn
from .module import SoLU


class SoLULayer(nn.Module):
    """
    The effective block used in the paper to recover performance:
    f(x) = LayerNorm(SoLU(x))
    """

    def __init__(self, hidden_size: int, dim=-1):
        super().__init__()
        self.solu = SoLU(dim=dim)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply SoLU first
        x = self.solu(x)
        # Apply LayerNorm immediately after
        return self.layer_norm(x)
