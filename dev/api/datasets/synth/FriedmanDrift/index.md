# FriedmanDrift

Friedman synthetic dataset with concept drifts.

Each observation is composed of 10 features. Each feature value is sampled uniformly in \[0, 1\]. Only the first 5 features are relevant. The target is defined by different functions depending on the type of the drift.

The three available modes of operation of the data generator are described in [^/dev/api/datasets/synth/FriedmanDrift/#1].

## Parameters

- **drift\_type**

  *Type* → `str`

  *Default* → `lea`

  The variant of concept drift. - `'lea'`: Local Expanding Abrupt drift. The concept drift appears in two distinct regions of the instance space, while the remaining regions are left unaltered. There are three points of abrupt change in the training dataset. At every consecutive change the regions of drift are expanded. - `'gra'`: Global Recurring Abrupt drift. The concept drift appears over the whole instance space. There are two points of concept drift. At the second point of drift the old concept reoccurs. - `'gsg'`: Global and Slow Gradual drift. The concept drift affects all the instance space. However, the change is gradual and not abrupt. After each one of the two change points covered by this variant, and during a window of length `transition_window`, examples from both old and the new concepts are generated with equal probability. After the transition period, only the examples from the new concept are generated.
- **position**

  *Type* → `tuple[int, ...]`

  *Default* → `(50000, 100000, 150000)`

  The amount of monitored instances after which each concept drift occurs. A tuple with at least two element must be passed, where each number is greater than the preceding one. If `drift_type='lea'`, then the tuple must have three elements.
- **transition\_window**

  *Type* → `int`

  *Default* → `10000`

  The length of the transition window between two concepts. Only applicable when `drift_type='gsg'`. If set to zero, the drifts will be abrupt. Anytime `transition_window > 0`, it defines a window in which instances of the new concept are gradually introduced among the examples from the old concept. During this transition phase, both old and new concepts appear with equal probability.
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

dataset = synth.FriedmanDrift(
    drift_type='lea',
    position=(1, 2, 3),
    seed=42
)

for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
[0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
[0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 7.04
[0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 18.16
[0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] -2.65
```

```python
dataset = synth.FriedmanDrift(
    drift_type='gra',
    position=(2, 3),
    seed=42
)

for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
[0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
[0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 8.96
[0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 18.16
[0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] 8.90
```

```python
dataset = synth.FriedmanDrift(
    drift_type='gsg',
    position=(1, 4),
    transition_window=2,
    seed=42
)

for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
[0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
[0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 8.92
[0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 17.32
[0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] 6.05
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

[^/dev/api/datasets/synth/FriedmanDrift/#1]: Ikonomovska, E., Gama, J. and Džeroski, S., 2011. Learning model trees from evolving data streams. Data mining and knowledge discovery, 23(1), pp.128-168.
