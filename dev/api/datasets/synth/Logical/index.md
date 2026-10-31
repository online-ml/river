# Logical

Logical functions stream generator.

Make a toy dataset with three labels that represent the logical functions: `OR`, `XOR`, `AND` (functions of the 2D input).

Data is generated in 'tiles' which contain the complete set of logical operations results. The tiles are repeated `n_tiles` times. Optionally, the generated data can be shuffled.

## Parameters

- **n\_tiles**

  *Type* → `int`

  *Default* → `1`

  Number of tiles to generate.
- **shuffle**

  *Type* → `bool`

  *Default* → `True`

  If set, generated data will be shuffled.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.Logical(n_tiles=2, shuffle=True, seed=42)

for x, y in dataset.take(5):
    print(x, y)
```

```text
{'A': 1, 'B': 1} {'OR': 1, 'XOR': 0, 'AND': 1}
{'A': 0, 'B': 0} {'OR': 0, 'XOR': 0, 'AND': 0}
{'A': 1, 'B': 0} {'OR': 1, 'XOR': 1, 'AND': 0}
{'A': 1, 'B': 1} {'OR': 1, 'XOR': 0, 'AND': 1}
{'A': 1, 'B': 0} {'OR': 1, 'XOR': 1, 'AND': 0}
```

## Methods

> [!NOTE]
>
> **take**
>
> Iterate over the k samples.
>
> **Parameters**
>
> - **k** — `int`
