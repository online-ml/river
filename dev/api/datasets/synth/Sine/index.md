# Sine

Sine generator.

This generator is an implementation of the dara stream with abrupt concept drift, as described in Gama, Joao, et al. [^/dev/api/datasets/synth/Sine/#1].

It generates up to 4 relevant numerical features, that vary from 0 to 1, where only 2 of them are relevant to the classification task and the other 2 are optionally added by as noise. A classification function is chosen among four options:

1. `SINE1`. Abrupt concept drift, noise-free examples. It has two relevant attributes. Each attributes has values uniformly distributed in \[0, 1\]. In the first context all points below the curve \\(y = sin(x)\\) are classified as positive.
2. `Reversed SINE1`. The reversed classification of `SINE1`.
3. `SINE2`. The same two relevant attributes. The classification function is \\(y \< 0.5 + 0.3 sin(3 \\pi x)\\).
4. `Reversed SINE2`. The reversed classification of `SINE2`.

Concept drift can be introduced by changing the classification function. This can be done manually or using `ConceptDriftStream`.

Two important features are the possibility to balance classes, which means the class distribution will tend to a uniform one, and the possibility to add noise, which will, add two non relevant attributes.

## Parameters

- **classification\_function**

  *Type* → `int`

  *Default* → `0`

  Classification functions to use. From 0 to 3.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **balance\_classes**

  *Type* → `bool`

  *Default* → `False`

  Whether to balance classes or not. If balanced, the class distribution will converge to an uniform distribution.
- **has\_noise**

  *Type* → `bool`

  *Default* → `False`

  Adds 2 non relevant features to the stream.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.Sine(classification_function = 2, seed = 112,
                     balance_classes = False, has_noise = True)

for x, y in dataset.take(5):
    print(x, y)
```

```text
{0: 0.4812, 1: 0.6660, 2: 0.6198, 3: 0.6994} 1
{0: 0.9022, 1: 0.7518, 2: 0.1625, 3: 0.2209} 0
{0: 0.4547, 1: 0.3901, 2: 0.9629, 3: 0.7287} 0
{0: 0.4683, 1: 0.3515, 2: 0.2273, 3: 0.6027} 0
{0: 0.9238, 1: 0.1673, 2: 0.4522, 3: 0.3447} 0
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

The sample generation works as follows: The two attributes are generated with the random number generator. The classification function defines whether to classify the instance as class 0 or class 1. Finally, data is balanced and noise is added, if these options are set by the user.

The generated sample will have 2 relevant features, and an additional two noise features if `has_noise` is set.

[^/dev/api/datasets/synth/Sine/#1]: Gama, Joao, et al.'s 'Learning with drift detection.' Advances in artificial intelligence-SBIA 2004. Springer Berlin Heidelberg, 2004. 286-295."
