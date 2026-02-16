import torch
import torch.nn as nn
import torch.nn.functional as F


class SoLU(nn.Module):
    """
    Softmax Linear Unit (SoLU) activation function.
    Formula: f(x) = x * softmax(x)
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applies softmax along the last dimension (typically the feature dimension in Transformers)
        return x * F.softmax(x, dim=self.dim)
