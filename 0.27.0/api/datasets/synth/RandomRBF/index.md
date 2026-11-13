# RandomRBF

Random Radial Basis Function generator.

Produces a radial basis function stream. A number of centroids, having a random central position, a standard deviation, a class label and weight are generated. A new sample is created by choosing one of the centroids at random, taking into account their weights, and offsetting the attributes in a random direction from the centroid's center. The offset length is drawn from a Gaussian distribution.

This process will create a normally distributed hypersphere of samples on the surrounds of each centroid.

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

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth
dataset = synth.RandomRBF(seed_model=42, seed_sample=42,
                          n_classes=4, n_features=4, n_centroids=20)
for x, y in dataset.take(5):
    print(x, y)
```

```text
{0: 1.0989, 1: 0.3840, 2: 0.7759, 3: 0.6592} 2
{0: 0.2366, 1: 1.3233, 2: 0.5691, 3: 0.2083} 0
{0: 1.3540, 1: -0.3306, 2: 0.1683, 3: 0.8865} 0
{0: 0.2585, 1: -0.2217, 2: 0.4739, 3: 0.6522} 0
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
