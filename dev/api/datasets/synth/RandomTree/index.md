# RandomTree

Random Tree generator.

This generator is based on [^/dev/api/datasets/synth/RandomTree/#1]. The generator creates a random tree by splitting features at random and setting labels at its leaves.

The tree structure is composed of node objects, which can be either inner nodes or leaf nodes. The choice comes as a function of the parameters passed to its initializer.

Since the concepts are generated and classified according to a tree structure, in theory, it should favor decision tree learners.

## Parameters

- **seed\_tree**

  *Type* → `int | None`

  *Default* → `None`

  Seed for random generation of tree.
- **seed\_sample**

  *Type* → `int | None`

  *Default* → `None`

  Seed for random generation of instances.
- **n\_classes**

  *Type* → `int`

  *Default* → `2`

  The number of classes to generate.
- **n\_num\_features**

  *Type* → `int`

  *Default* → `5`

  The number of numerical features to generate.
- **n\_cat\_features**

  *Type* → `int`

  *Default* → `5`

  The number of categorical features to generate.
- **n\_categories\_per\_feature**

  *Type* → `int`

  *Default* → `5`

  The number of values to generate per categorical feature.
- **max\_tree\_depth**

  *Type* → `int`

  *Default* → `5`

  The maximum depth of the tree concept.
- **first\_leaf\_level**

  *Type* → `int`

  *Default* → `3`

  The first level of the tree above `max_tree_depth` that can have leaves.
- **fraction\_leaves\_per\_level**

  *Type* → `float`

  *Default* → `0.15`

  The fraction of leaves per level from `first_leaf_level` onwards.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.RandomTree(seed_tree=42, seed_sample=42, n_classes=2,
                           n_num_features=2, n_cat_features=2,
                           n_categories_per_feature=2, max_tree_depth=6,
                           first_leaf_level=3, fraction_leaves_per_level=0.15)

for x, y in dataset.take(5):
    print(x, y)
```

```text
{'x_num_0': 0.6394, 'x_num_1': 0.0250, 'x_cat_0': 1, 'x_cat_1': 0} 0
{'x_num_0': 0.2232, 'x_num_1': 0.7364, 'x_cat_0': 0, 'x_cat_1': 1} 1
{'x_num_0': 0.0317, 'x_num_1': 0.0936, 'x_cat_0': 0, 'x_cat_1': 0} 0
{'x_num_0': 0.5612, 'x_num_1': 0.7160, 'x_cat_0': 1, 'x_cat_1': 0} 0
{'x_num_0': 0.4492, 'x_num_1': 0.2781, 'x_cat_0': 0, 'x_cat_1': 0} 0
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

[^/dev/api/datasets/synth/RandomTree/#1]: Domingos, Pedro, and Geoff Hulten. "Mining high-speed data streams." In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 71-80. 2000.
