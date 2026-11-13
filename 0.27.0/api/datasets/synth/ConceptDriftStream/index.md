# ConceptDriftStream

Generates a stream with concept drift.

A stream generator that adds concept drift or change by joining two streams. This is done by building a weighted combination of two pure distributions that characterizes the target concepts before and after the change.

The sigmoid function is an elegant and practical solution to define the probability that each new instance of the stream belongs to the new concept after the drift. The sigmoid function introduces a gradual, smooth transition whose duration is controlled with two parameters:

- \\(p\\), the position of the change.
- \\(w\\), the width of the transition.

The sigmoid function at sample \\(t\\) is

\\\[f(t) = 1/(1+e^{-4(t-p)/w})\\\]

## Parameters

- **stream**

  *Type* → `datasets.base.SyntheticDataset | None`

  *Default* → `None`

  Original stream
- **drift\_stream**

  *Type* → `datasets.base.SyntheticDataset | None`

  *Default* → `None`

  Drift stream
- **position**

  *Type* → `int`

  *Default* → `5000`

  Central position of the concept drift change.
- **width**

  *Type* → `int`

  *Default* → `1000`

  Width of concept drift change.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **alpha**

  *Type* → `float | None`

  *Default* → `None`

  Angle of change used to estimate the width of concept drift change. If set, it will override the width parameter. Valid values are in the range (0.0, 90.0\].

## Attributes

- **desc**

  Return the description from the docstring.

## Examples

```python
from river.datasets import synth

dataset = synth.ConceptDriftStream(
    stream=synth.SEA(seed=42, variant=0),
    drift_stream=synth.SEA(seed=42, variant=1),
    seed=1, position=5, width=2
)

for x, y in dataset.take(10):
    print(x, y)
```

```text
{0: 6.3942, 1: 0.2501, 2: 2.7502} False
{0: 2.2321, 1: 7.3647, 2: 6.7669} True
{0: 8.9217, 1: 0.8693, 2: 4.2192} True
{0: 0.2979, 1: 2.1863, 2: 5.0535} False
{0: 6.3942, 1: 0.2501, 2: 2.7502} False
{0: 2.2321, 1: 7.3647, 2: 6.7669} True
{0: 8.9217, 1: 0.8693, 2: 4.2192} True
{0: 0.2979, 1: 2.1863, 2: 5.0535} False
{0: 0.2653, 1: 1.9883, 2: 6.4988} False
{0: 5.4494, 1: 2.2044, 2: 5.8926} False
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

An optional way to estimate the width of the transition \\(w\\) is based on the angle \\(\\alpha\\), \\(w = 1/ tan(\\alpha)\\). Since width corresponds to the number of samples for the transition, the width is rounded to the nearest smaller integer. Notice that larger values of \\(\\alpha\\) result in smaller widths. For \\(\\alpha \> 45.0\\), the width is smaller than 1 so values are rounded to 1 to avoid division by zero errors.
