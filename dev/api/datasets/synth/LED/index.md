# LED

LED stream generator.

This data source originates from the CART book [^/dev/api/datasets/synth/LED/#1]. An implementation in C was donated to the UCI [^/dev/api/datasets/synth/LED/#2] machine learning repository by David Aha. The goal is to predict the digit displayed on a seven-segment LED display, where each attribute has a 10% chance of being inverted. It has an optimal Bayes classification rate of 74%. The particular configuration of the generator used for experiments (LED) produces 24 binary attributes, 17 of which are irrelevant.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **noise\_percentage**

  *Type* → `float`

  *Default* → `0.0`

  The probability that noise will happen in the generation. At each new sample generated, a random number is generated, and if it is equal or less than the noise\_percentage, the led value will be switched
- **irrelevant\_features**

  *Type* → `bool`

  *Default* → `False`

  Adds 17 non-relevant attributes to the stream.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.LED(seed = 112, noise_percentage = 0.28, irrelevant_features= False)

for x, y in dataset.take(5):
    print(x, y)
```

```text
{0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0} 7
{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0} 8
{0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0} 9
{0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0} 1
{0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0} 1
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

An instance is generated based on the parameters passed. If `has_noise` is set then the total number of attributes will be 24, otherwise there will be 7 attributes.

[^/dev/api/datasets/synth/LED/#1]: Leo Breiman, Jerome Friedman, R. Olshen, and Charles J. Stone. Classification and Regression Trees. Wadsworth and Brooks, Monterey, CA,1984.

[^/dev/api/datasets/synth/LED/#2]: A. Asuncion and D. J. Newman. UCI Machine Learning Repository \[http://www.ics.uci.edu/\~mlearn/mlrepository.html\]. University of California, Irvine, School of Information and Computer Sciences,2007.
