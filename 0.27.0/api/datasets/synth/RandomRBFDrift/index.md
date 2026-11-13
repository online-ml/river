# RandomRBFDrift

Random Radial Basis Function generator with concept drift.

This class is an extension from the `RandomRBF` generator. Concept drift can be introduced in instances of this class.

The drift is created by adding a "speed" to certain centroids. As the samples are generated each of the moving centroids' centers is changed by an amount determined by its speed.

## Parameters

- **seed\_model**

  *Type* → `int | None`

  *Default* → `None`

  Model's random seed to generate centroids.
- **seed\_sample**

  *Type* → `int | None`

  *Default* → `None`

  Sample's random seed.
- **n\_classes**

  *Type* → `int`

  *Default* → `2`

  The number of class labels to generate.
- **n\_features**

  *Type* → `int`

  *Default* → `10`

  The number of numerical features to generate.
- **n\_centroids**

  *Type* → `int`

  *Default* → `50`

  The number of centroids to generate.
- **change\_speed**

  *Type* → `float`

  *Default* → `0.0`

  The concept drift speed.
- **n\_drift\_centroids**

  *Type* → `int`

  *Default* → `50`

  The number of centroids that will drift.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth
dataset = synth.RandomRBFDrift(seed_model=42, seed_sample=42,
                               n_classes=4, n_features=4, n_centroids=20,
                               change_speed=0.87, n_drift_centroids=10)
for x, y in dataset.take(5):
    print(x, y)
```

```text
{0: 1.0989, 1: 0.3840, 2: 0.7759, 3: 0.6592} 2
{0: 1.1496, 1: 1.9014, 2: 1.5393, 3: 0.3210} 0
{0: 0.7146, 1: -0.2414, 2: 0.8933, 3: 1.6633} 0
{0: 0.3797, 1: -0.1027, 2: 0.8717, 3: 1.1635} 0
{0: 0.1295, 1: 0.5953, 2: 0.1774, 3: 0.6673} 1
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
