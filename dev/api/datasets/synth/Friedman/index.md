# Friedman

Friedman synthetic dataset.

Each observation is composed of 10 features. Each feature value is sampled uniformly in \[0, 1\]. The target is defined by the following function:

\\\[y = 10 sin(\\pi x\_0 x\_1) + 20 (x\_2 - 0.5)^2 + 10 x\_3 + 5 x\_4 + \\epsilon\\\]

In the last expression, \\(\\epsilon \\sim \\mathcal{N}(0, 1)\\), is the noise. Therefore, only the first 5 features are relevant.

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

dataset = synth.Friedman(seed=42)

for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
[0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
[0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 7.04
[0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 18.16
[0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] 8.90
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

## References

- [Friedman, J.H., 1991. Multivariate adaptive regression splines. The annals of statistics, pp.1-67.](<https://projecteuclid.org/euclid.aos/1176347963>)
