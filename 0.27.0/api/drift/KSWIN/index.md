# KSWIN

Kolmogorov-Smirnov Windowing method for concept drift detection.

## Parameters

- **alpha**

  *Type* → `float`

  *Default* → `0.005`

  Probability for the test statistic of the Kolmogorov-Smirnov-Test. The alpha parameter is very sensitive, therefore should be set below 0.01.
- **window\_size**

  *Type* → `int`

  *Default* → `100`

  Size of the sliding window.
- **stat\_size**

  *Type* → `int`

  *Default* → `30`

  Size of the statistic window.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.
- **window**

  *Type* → `typing.Iterable[int | float] | None`

  *Default* → `None`

  Already collected data to avoid cold start.

## Attributes

- **drift\_detected**

  Whether or not a drift is detected following the last update.

## Examples

```python
import random
from river import drift

rng = random.Random(12345)
kswin = drift.KSWIN(alpha=0.0001, seed=42)

data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

for i, val in enumerate(data_stream):
    kswin.update(val)
    if kswin.drift_detected:
        print(f"Change detected at index {i}, input value: {val}")
```

```text
Change detected at index 1016, input value: 6
```

## Methods

> [!NOTE]
>
> **update**
>
> Update the change detector with a single data point.
>
> Adds an element on top of the sliding window and removes the oldest one from the window. Afterwards, the KS-test is performed.
>
> **Parameters**
>
> - **x** — `int | float`

## Notes

KSWIN (Kolmogorov-Smirnov Windowing) is a concept change detection method based on the Kolmogorov-Smirnov (KS) statistical test. KS-test is a statistical test with no assumption of underlying data distribution. KSWIN can monitor data or performance distributions. Note that the detector accepts one dimensional input as array.

KSWIN maintains a sliding window \\(\\Psi\\) of fixed size \\(n\\) (window\_size). The last \\(r\\) (stat\_size) samples of \\(\\Psi\\) are assumed to represent the last concept considered as \\(R\\). From the first \\(n-r\\) samples of \\(\\Psi\\), \\(r\\) samples are uniformly drawn, representing an approximated last concept \\(W\\).

The KS-test is performed on the windows \\(R\\) and \\(W\\) of the same size. KS -test compares the distance of the empirical cumulative data distribution \\(dist(R,W)\\).

A concept drift is detected by KSWIN if:

\\\[ dist(R,W) \> \\sqrt{-\\frac{ln\\alpha}{r}} \\\]

The difference in empirical data distributions between the windows \\(R\\) and \\(W\\) is too large since \\(R\\) and \\(W\\) come from the same distribution.

## References

- Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,
