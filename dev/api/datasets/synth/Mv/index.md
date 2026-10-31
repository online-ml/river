# Mv

Mv artificial dataset.

Artificial dataset composed of both nominal and numeric features, whose features present co-dependencies. Originally described in [^/dev/api/datasets/synth/Mv/#1].

The features are generated using the following expressions:

- \\(x\_1\\): uniformly distributed over `\[-5, 5\]`.
- \\(x\_2\\): uniformly distributed over `\[-15, -10\]`.
- \\(x\_3\\):

  - if \\(x\_1 \> 0\\), \\(x\_3 \\leftarrow\\) `'green'`
  - else \\(x\_3 \\leftarrow\\) `'red'` with probability \\(0.4\\) and \\(x\_3 \\leftarrow\\) `'brown'` with probability \\(0.6\\).
- \\(x\_4\\):

  - if \\(x\_3 =\\) `'green'`, \\(x\_4 \\leftarrow x\_1 + 2 x\_2\\)
  - else \\(x\_4 = \\frac{x\_1}{2}\\) with probability \\(0.3\\) and \\(x\_4 = \\frac{x\_2}{2}\\) with probability \\(0.7\\).
- \\(x\_5\\): uniformly distributed over `\[-1, 1\]`.
- \\(x\_6 \\leftarrow x\_4 \\times \\epsilon\\), where \\(\\epsilon\\) is uniformly distributed

over `\[0, 5\]`.

- \\(x\_7\\): `'yes'` with probability \\(0.3\\), and `'no'` with probability \\(0.7\\).
- \\(x\_8\\): `'normal'` if \\(x\_5 \< 0.5\\) else `'large'`.
- \\(x\_9\\): uniformly distributed over `\[100, 500\]`.
- \\(x\_{10}\\): uniformly distributed integer over the interval `\[1000, 1200\]`.

The target value is generated using the following rules:

- if \\(x\_2 \> 2\\), \\(y \\leftarrow 35 - 0.5 x\_4\\)
- else if \\(-2 \\le x\_4 \\le 2\\), \\(y \\leftarrow 10 - 2 x\_1\\)
- else if \\(x\_7 =\\) `'yes'`, \\(y \\leftarrow 3 - \\frac{x\_1}{x\_4}\\)
- else if \\(x\_8 =\\) `'normal'`, \\(y \\leftarrow x\_6 + x\_1\\)
- else \\(y \\leftarrow \\frac{x\_1}{2}\\).

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

dataset = synth.Mv(seed=42)

for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[1.39, -14.87, 'green', -28.35, -0.44, -31.64, 'no', 'normal', 370.67, 1178.43] -30.25
[-4.13, -12.89, 'red', -2.06, 0.01, -0.27, 'yes', 'normal', 359.95, 1108.98] 1.00
[-2.79, -12.05, 'brown', -1.39, 0.61, -4.87, 'no', 'large', 162.19, 1191.44] 15.59
[-1.63, -14.53, 'red', -7.26, 0.20, -29.33, 'no', 'normal', 314.49, 1194.62] -30.96
[-1.21, -12.23, 'brown', -6.11, 0.72, -17.66, 'no', 'large', 118.32, 1045.57] -0.60
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

[^/dev/api/datasets/synth/Mv/#1]: [Mv in Luís Torgo regression datasets](<https://www.dcc.fc.up.pt/~ltorgo/Regression/mv.html>)
