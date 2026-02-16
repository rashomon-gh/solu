# solu - Softmax Linear Unit

This repository packages an implementation of Sofmax Linear Unit, as proposed in [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html#section-3-2).

## Module Structure

```
solu -> SoLU, SoLULayer
```


## Performance Penalty Mitigation

The original paper talks about a performance penalty with softmax linear unit which can be mitigated with an additional Layer Norm. This mitigation has been applied in the `SoLULayer` module in this package. The activation function itself is in the `SoLU` module. 

## Example Usage

> [!NOTE]
> `SoLU` and `SoLULayer` are `torch.nn` modules and hence can be used in any pytorch model definition.


```python
import torch
from solu import SoLULayer, SoLU


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
```

****You can also check `main.py`****



