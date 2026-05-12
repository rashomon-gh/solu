"""Core SoLU activation module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoLU(nn.Module):
    """Softmax Linear Unit (SoLU) activation function.

    The SoLU activation function applies a softmax operation along a specified
    dimension and multiplies it element-wise with the input tensor:

        f(x) = x * softmax(x, dim=dim)

    This activation function creates a multiplicative interaction between the
    input and its normalized version, which can help with gradient flow and
    feature learning in deep neural networks.

    Attributes:
        dim: The dimension along which to apply softmax.

    Example:
        >>> import torch
        >>> from SoLU import SoLU
        >>> solu = SoLU(dim=-1)
        >>> x = torch.randn(2, 5, 4)
        >>> output = solu(x)
        >>> assert output.shape == x.shape
    """

    def __init__(self, dim: int = -1):
        """Initialize the SoLU activation function.

        Args:
            dim: The dimension along which to apply softmax. Defaults to -1
                (the last dimension), which is typically the feature dimension
                in transformer architectures.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SoLU activation to the input tensor.

        Args:
            x: Input tensor of any shape. The softmax operation will be
                applied along the dimension specified in ``self.dim``.

        Returns:
            A tensor with the same shape as ``x``, where each element is
            the product of the corresponding input element and its softmax
            normalization along the specified dimension.
        """
        return x * F.softmax(x, dim=self.dim)
