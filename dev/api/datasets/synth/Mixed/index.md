# Mixed

Mixed data stream generator.

This generator is an implementation of a data stream with abrupt concept drift and boolean noise-free examples as described in [^/dev/api/datasets/synth/Mixed/#1].

It has four relevant attributes, two boolean attributes \\(v, w\\) and two numeric attributes \\(x, y\\) uniformly distributed from 0 to 1. The examples are labeled depending on the classification function chosen from below.

- `function 0`: if \\(v\\) and \\(w\\) are true or \\(v\\) and \\(z\\) are true or \\(w\\) and \\(z\\) are true then 0 else 1, where \\(z\\) is \\(y \< 0.5 + 0.3 sin(3 \\pi x)\\)
- `function 1`: The opposite of `function 0`.

Concept drift can be introduced by changing the classification function. This can be done manually or using `ConceptDriftStream`.

## Parameters

- **classification\_function**

  *Type* → `int`

  *Default* → `0`

  Which of the two classification functions to use for the generation. Valid options are 0 or 1.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **balance\_classes**

  *Type* → `bool`

  *Default* → `False`

  Whether to balance classes or not. If balanced, the class distribution will converge to a uniform distribution.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth
dataset = synth.Mixed(seed = 42, classification_function=1, balance_classes = True)
for x, y in dataset.take(5):
    print(x, y)
```

```text
{0: True, 1: False, 2: 0.2750, 3: 0.2232} 1
{0: False, 1: False, 2: 0.2186, 3: 0.5053} 0
{0: False, 1: True, 2: 0.8094, 3: 0.0064} 1
{0: False, 1: False, 2: 0.1010, 3: 0.2779} 0
{0: True, 1: False, 2: 0.37018, 3: 0.2095} 1
```

## Methods

> [!NOTE]
>
> **generate\_drift**
>
> Generate drift by switching the classification function.

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

The sample generation works as follows: The two numeric attributes are generated with the random generator initialized with the seed passed by the user (optional). The boolean attributes are either 0 or 1 based on the comparison of the random number generator and 0.5, the classification function decides whether to classify the instance as class 0 or class 1. The next step is to verify if the classes should be balanced, and if so, balance the classes.

The generated sample will have 4 relevant features and 1 label (it is a binary-classification task).

[^/dev/api/datasets/synth/Mixed/#1]: Gama, Joao, et al. "Learning with drift detection." Advances in artificial intelligence-SBIA 2004. Springer Berlin Heidelberg, 2004. 286-295"
