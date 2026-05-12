"""Demo script for testing the SoLU module.

This script demonstrates basic usage of the SoLU and SoLULayer modules
with a simple forward pass example.
"""

import torch

from SoLU import SoLULayer


@torch.no_grad()
def main() -> None:
    """Run a demo forward pass through the SoLULayer.

    Creates a random input tensor and passes it through a SoLULayer
    to demonstrate the module's functionality. The output tensor and
    its shape are printed to stdout.
    """
    # batch_size=2, seq_len=5, hidden_dim=4
    x = torch.randn(2, 5, 4)

    # Initialize the layer (SoLU + LayerNorm)
    solu_block = SoLULayer(hidden_size=4)

    # Forward Pass
    output = solu_block(x)
    print(output)
    print(output.size())


if __name__ == "__main__":
    main()
