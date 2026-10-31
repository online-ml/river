# Agrawal

Agrawal stream generator.

The generator was introduced by Agrawal et al. [^/dev/api/datasets/synth/Agrawal/#1], and was a common source of data for early work on scaling up decision tree learners. The generator produces a stream containing nine features, six numeric and three categorical. There are 10 functions defined for generating binary class labels from the features. Presumably these determine whether the loan should be approved. Classification functions are listed in the original paper [^/dev/api/datasets/synth/Agrawal/#1].

**Feature** | **Description** | **Values**

- `salary` | salary | uniformly distributed from 20k to 150k
- `commission` | commission | 0 if `salary` \< 75k else uniformly distributed from 10k to 75k
- `age` | age | uniformly distributed from 20 to 80
- `elevel` | education level | uniformly chosen from 0 to 4
- `car` | car maker | uniformly chosen from 1 to 20
- `zipcode` | zip code of the town | uniformly chosen from 0 to 8
- `hvalue` | house value | uniformly distributed from 50k x zipcode to 100k x zipcode
- `hyears` | years house owned | uniformly distributed from 1 to 30
- `loan` | total loan amount | uniformly distributed from 0 to 500k

## Parameters

- **classification\_function**

  *Type* → `int`

  *Default* → `0`

  The classification function to use for the generation. Valid values are from 0 to 9.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **balance\_classes**

  *Type* → `bool`

  *Default* → `False`

  If True, the class distribution will converge to a uniform distribution.
- **perturbation**

  *Type* → `float`

  *Default* → `0.0`

  The probability that noise will happen in the generation. Each new sample will be perturbed by the magnitude of `perturbation`. Valid values are in the range \[0.0 to 1.0\].

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.Agrawal(
    classification_function=0,
    seed=42
)

dataset
```

```text
Synthetic data generator
<BLANKLINE>
    Name  Agrawal
    Task  Binary classification
 Samples  ∞
Features  9
 Outputs  1
 Classes  2
  Sparse  False
<BLANKLINE>
Configuration
-------------
classification_function  0
                   seed  42
        balance_classes  False
           perturbation  0.0
```

```python
for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[103125.4837, 0, 21, 2, 8, 3, 319768.9642, 4, 338349.7437] 1
[135983.3438, 0, 25, 4, 14, 0, 423837.7755, 7, 116330.4466] 1
[98262.4347, 0, 55, 1, 18, 6, 144088.1244, 19, 139095.3541] 0
[133009.0417, 0, 68, 1, 14, 5, 233361.4025, 7, 478606.5361] 1
[63757.2908, 16955.9382, 26, 2, 12, 4, 522851.3093, 24, 229712.4398] 1
```

## Methods

> [!NOTE]
>
> **generate\_drift**
>
> Generate drift by switching the classification function randomly.

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

The sample generation works as follows: The 9 features are generated with the random generator, initialized with the seed passed by the user. Then, the classification function decides, as a function of all the attributes, whether to classify the instance as class 0 or class 1. The next step is to verify if the classes should be balanced, and if so, balance the classes. Finally, add noise if `perturbation` \> 0.0.

[^/dev/api/datasets/synth/Agrawal/#1]: Rakesh Agrawal, Tomasz Imielinksi, and Arun Swami. "Database Mining: A Performance Perspective", IEEE Transactions on Knowledge and Data Engineering, 5(6), December 1993.
