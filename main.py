import torch
from SoLU import SoLULayer


@torch.no_grad()
def main():
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
