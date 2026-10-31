# Hyperplane

Hyperplane stream generator.

Generates a problem of prediction class of a rotation hyperplane. It was used as testbed for CVFDT and VFDT in [^/dev/api/datasets/synth/Hyperplane/#1].

A hyperplane in d-dimensional space is the set of points \\(x\\) that satisfy

\\\[\\sum^{d}\_{i=1} w\_i x\_i = w\_0 = \\sum^{d}\_{i=1} w\_i\\\]

where \\(x\_i\\) is the i-th coordinate of \\(x\\).

- Examples for which \\(\\sum^{d}\_{i=1} w\_i x\_i \> w\_0\\), are labeled positive.
- Examples for which \\(\\sum^{d}\_{i=1} w\_i x\_i \\leq w\_0\\), are labeled negative.

Hyperplanes are useful for simulating time-changing concepts because we can change the orientation and position of the hyperplane in a smooth manner by changing the relative size of the weights. We introduce change to this dataset by adding drift to each weighted feature \\(w\_i = w\_i + d \\sigma\\), where \\(\\sigma\\) is the probability that the direction of change is reversed and \\(d\\) is the change applied to each example.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **n\_features**

  *Type* → `int`

  *Default* → `10`

  The number of attributes to generate. Higher than 2.
- **n\_drift\_features**

  *Type* → `int`

  *Default* → `2`

  The number of attributes with drift. Higher than 2.
- **mag\_change**

  *Type* → `float`

  *Default* → `0.0`

  Magnitude of the change for every example. From 0.0 to 1.0.
- **noise\_percentage**

  *Type* → `float`

  *Default* → `0.05`

  Percentage of noise to add to the data. From 0.0 to 1.0.
- **sigma**

  *Type* → `float`

  *Default* → `0.1`

  Probability that the direction of change is reversed. From 0.0 to 1.0.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.Hyperplane(seed=42, n_features=2)

for x, y in dataset.take(5):
    print(x, y)
```

```text
{0: 0.2750, 1: 0.2232} 0
{0: 0.0869, 1: 0.4219} 1
{0: 0.0265, 1: 0.1988} 0
{0: 0.5892, 1: 0.8094} 0
{0: 0.3402, 1: 0.1554} 0
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

## Notes

The sample generation works as follows: The features are generated with the random number generator, initialized with the seed passed by the user. Then the classification function decides, as a function of the sum of the weighted features and the sum of the weights, whether the instance belongs to class 0 or class 1. The last step is to add noise and generate drift.

[^/dev/api/datasets/synth/Hyperplane/#1]: G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams. In KDD'01, pages 97-106, San Francisco, CA, 2001. ACM Press.
