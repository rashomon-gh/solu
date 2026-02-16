# SoLU - Softmax Linear Unit

This repository packages an implementation of Sofmax Linear Unit, as proposed in [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html#section-3-2).

## Module Structure

```
SoLU -> SoLU, SoLULayer
```


## Performance Penalty Mitigation

The original paper talks about a performance penalty with softmax linear unit which can be mitigated with an additional Layer Norm. This mitigation has been applied in the `SoLULayer` module in this package. The activation function itself is in the `SoLU` module. 

## Example Usage

### Installation

```bash
pip install softmax-linear-unit
```

### Code import

> [!NOTE]
> `SoLU` and `SoLULayer` are `torch.nn` modules and hence can be used in any pytorch model definition.


```python
import torch
from SoLU import SoLULayer, SoLU


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


## Local Dev

### Env

```bash
# make sure to have uv installed
# also python 3.12.11

uv sync
source .venv/bin/activate
```

### Ruff and Pre-Commit

By default, `pre-commit` will run `ruff` formatting with the `--fix` flag.


> [!NOTE]
> The pre-commit configuration can be found in the `.pre-commit-config.yaml` file.

```bash
pre-commit install
```

