# Planes2D

2D Planes synthetic dataset.

This dataset is described in [^/dev/api/datasets/synth/Planes2D/#1] and was adapted from [^/dev/api/datasets/synth/Planes2D/#2]. The features are generated using the following probabilities:

\\\[P(x\_1 = -1) = P(x\_1 = 1) = \\frac{1}{2}\\\]

\\\[P(x\_m = -1) = P(x\_m = 0) = P(x\_m = 1) = \\frac{1}{3}, m=2,\\ldots, 10\\\]

The target value is defined by the following rule:

\\\[\\text{if}\~x\_1 = 1, y \\leftarrow 3 + 3x\_2 + 2x\_3 + x\_4 + \\epsilon\\\]

\\\[\\text{if}\~x\_1 = -1, y \\leftarrow -3 + 3x\_5 + 2x\_6 + x\_7 + \\epsilon\\\]

In the expressions, \\(\\epsilon \\sim \\mathcal{N}(0, 1)\\), is the noise.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed number used for reproducibility.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.Planes2D(seed=42)

for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[-1, -1, 1, 0, -1, -1, -1, 1, -1, 1] -9.07
[1, -1, -1, -1, -1, -1, 1, 1, -1, 1] -4.25
[-1, 1, 1, 1, 1, 0, -1, 0, 1, 0] -0.95
[-1, 1, 0, 0, 0, -1, -1, 0, -1, -1] -6.10
[1, -1, 0, 0, 1, 0, -1, 1, 0, 1] 1.60
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

[^/dev/api/datasets/synth/Planes2D/#1]: [2DPlanes in Luís Torgo regression datasets](<https://www.dcc.fc.up.pt/~ltorgo/Regression/2dplanes.html>)

[^/dev/api/datasets/synth/Planes2D/#2]: Breiman, L., Friedman, J., Stone, C.J. and Olshen, R.A., 1984. Classification and regression trees. CRC press.
