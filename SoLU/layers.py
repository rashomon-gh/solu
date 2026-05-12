"""Neural network layers incorporating SoLU activation."""

import torch
import torch.nn as nn
from .module import SoLU


class SoLULayer(nn.Module):
    """A neural network layer combining SoLU activation with LayerNorm.

    This layer implements the effective block used in recent research to
    recover and improve performance in transformer-like architectures:

        f(x) = LayerNorm(SoLU(x))

    The combination of SoLU activation followed by LayerNormalization
    provides stable training dynamics and has been shown to improve
    convergence in deep networks.

    Attributes:
        solu: The SoLU activation function module.
        layer_norm: A LayerNorm module that normalizes across the hidden size.

    Example:
        >>> import torch
        >>> from SoLU import SoLULayer
        >>> layer = SoLULayer(hidden_size=4)
        >>> x = torch.randn(2, 5, 4)  # batch_size=2, seq_len=5, hidden_dim=4
        >>> output = layer(x)
        >>> assert output.shape == x.shape
    """

    def __init__(self, hidden_size: int, dim: int = -1):
        """Initialize the SoLULayer.

        Args:
            hidden_size: The size of the hidden dimension, which determines
                the shape normalization for LayerNorm.
            dim: The dimension along which to apply softmax in the SoLU
                activation. Defaults to -1 (the last dimension).
        """
        super().__init__()
        self.solu = SoLU(dim=dim)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SoLU activation and LayerNorm to the input tensor.

        Args:
            x: Input tensor of shape ``(*, hidden_size)`` where ``hidden_size``
                matches the ``hidden_size`` passed to the constructor.

        Returns:
            A tensor with the same shape as ``x``, after applying SoLU
            activation followed by LayerNormalization.
        """
        x = self.solu(x)
        return self.layer_norm(x)
