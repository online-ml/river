# LEDDrift

LED stream generator with concept drift.

This class is an extension of the `LED` generator whose purpose is to add concept drift to the stream.

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
- **n\_drift\_features**

  *Type* → `int`

  *Default* → `0`

  The number of attributes that have drift.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.LEDDrift(seed = 112, noise_percentage = 0.28,
                         irrelevant_features= True, n_drift_features=4)

for x, y in dataset.take(5):
    print(list(x.values()), y)
```

```text
[1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1] 7
[1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] 6
[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1] 1
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1] 6
[1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0] 7
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
