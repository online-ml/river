# STAGGER

STAGGER concepts stream generator.

This generator is an implementation of the dara stream with abrupt concept drift, as described in [^/dev/api/datasets/synth/STAGGER/#1].

The STAGGER concepts are boolean functions `f` with three features describing objects: size (small, medium and large), shape (circle, square and triangle) and colour (red, blue and green).

`f` options:

1. `True` if the size is small and the color is red.
2. `True` if the color is green or the shape is a circle.
3. `True` if the size is medium or large

Concept drift can be introduced by changing the classification function. This can be done manually or using [`datasets.synth.ConceptDriftStream`](<https://riverml.xyz/dev/api/datasets/synth/ConceptDriftStream>).

One important feature is the possibility to balance classes, which means the class distribution will tend to a uniform one.

## Parameters

- **classification\_function**

  *Type* → `int`

  *Default* → `0`

  Classification functions to use. From 0 to 2.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **balance\_classes**

  *Type* → `bool`

  *Default* → `False`

  Whether to balance classes or not. If balanced, the class distribution will converge to an uniform distribution.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.STAGGER(classification_function = 2, seed = 112,
                     balance_classes = False)

for x, y in dataset.take(5):
    print(x, y)
```

```text
{'size': 1, 'color': 2, 'shape': 2} 1
{'size': 2, 'color': 1, 'shape': 2} 1
{'size': 1, 'color': 1, 'shape': 2} 1
{'size': 0, 'color': 1, 'shape': 0} 0
{'size': 2, 'color': 1, 'shape': 0} 1
```

## Methods

> [!NOTE]
>
> **generate\_drift**
>
> Generate drift by switching the classification function at random.

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

The sample generation works as follows: The 3 attributes are generated with the random number generator. The classification function defines whether to classify the instance as class 0 or class 1. Finally, data is balanced, if this option is set by the user.

[^/dev/api/datasets/synth/STAGGER/#1]: Schlimmer, J. C., &amp; Granger, R. H. (1986). Incremental learning from noisy data. Machine learning, 1(3), 317-354.
