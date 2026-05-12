"""Softmax Linear Unit (SoLU) - A PyTorch implementation.

This package implements the Softmax Linear Unit activation function
as described in https://www.anthropic.com/research/softmax-linear-units. SoLU applies
a softmax operation element-wise with the input, creating a unique
activation pattern that has been shown to improve training dynamics
in certain neural network architectures.

Example:
    >>> import torch
    >>> from SoLU import SoLU, SoLULayer
    >>> x = torch.randn(2, 5, 4)
    >>> solu = SoLU()
    >>> output = solu(x)
    >>> layer = SoLULayer(hidden_size=4)
    >>> output = layer(x)
"""

from .module import SoLU
from .layers import SoLULayer

__all__ = ["SoLU", "SoLULayer"]
