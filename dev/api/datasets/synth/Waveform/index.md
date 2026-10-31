# Waveform

Waveform stream generator.

Generates samples with 21 numeric features and 3 classes, based on a random differentiation of some base waveforms. Supports noise addition, in this case the samples will have 40 features.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **has\_noise**

  *Type* → `bool`

  *Default* → `False`

  Adds 19 unrelated features to the stream.

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.Waveform(seed=42, has_noise=True)

for x, y in dataset:
    break

x
```

```text
{0: -0.0397, 1: -0.7484, 2: 0.2974, 3: 0.3574, 4: -0.0735, 5: -0.3647, 6: 1.5631,     7: 2.5291, 8: 4.1599, 9: 4.9587, 10: 4.52587, 11: 4.0097, 12: 3.6705, 13: 1.7033,     14: 1.4898, 15: 1.9743, 16: 0.0898, 17: 2.319, 18: 0.2552, 19: -0.4775, 20: -0.71339,     21: 0.3770, 22: 0.3671, 23: 1.6579, 24: 0.7828, 25: 0.5855, 26: -0.5807, 27: 0.7112,     28: -0.0271, 29: 0.2968, 30: -0.4997, 31: 0.1302, 32: 0.3578, 33: -0.1900, 34: -0.3771,     35: 1.3560, 36: 0.7124, 37: -0.6245, 38: 0.1346, 39: 0.3550}
```

```python
y
```

```text
2
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

An instance is generated based on the parameters passed. The generator will randomly choose one of the hard coded waveforms, as well as random multipliers. For each feature, the actual value generated will be a a combination of the hard coded functions, with the multipliers and a random value.

If noise is added then the features 21 to 40 will be replaced with a random normal value.
