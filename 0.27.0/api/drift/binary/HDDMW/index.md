# HDDMW

Drift Detection Method based on Hoeffding's bounds with moving weighted average-test.

HDDMW is an online drift detection method based on McDiarmid's bounds. HDDMW uses the Exponentially Weighted Moving Average (EWMA) statistic as estimator.

**Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0 represents correct/normal values.

For example, if a classifier's prediction \\(y'\\) is right or wrong w.r.t. the true target label \\(y\\):

- 0: Correct, \\(y=y'\\)
- 1: Error, \\(y \\neq y'\\)

*Implementation based on MOA.*

## Parameters

- **drift\_confidence**

  *Type* → `float`

  *Default* → `0.001`

  Confidence to the drift
- **warning\_confidence**

  *Type* → `float`

  *Default* → `0.005`

  Confidence to the warning
- **lambda\_val**

  *Type* → `float`

  *Default* → `0.05`

  The weight given to recent data. Smaller values mean less weight given to recent data.
- **two\_sided\_test**

  *Type* → `bool`

  *Default* → `False`

  If True, will monitor error increments and decrements (two-sided). By default will only monitor increments (one-sided).

## Attributes

- **drift\_detected**

  Whether or not a drift is detected following the last update.
- **warning\_detected**

  Whether or not a drift is detected following the last update.

## Examples

```python
import random
from river import drift

rng = random.Random(42)
hddm_w = drift.binary.HDDMW()

data_stream = rng.choices([0, 1], k=1000)
data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])

print_warning = True
for i, x in enumerate(data_stream):
    hddm_w.update(x)
    if hddm_w.warning_detected and print_warning:
        print(f"Warning detected at index {i}")
        print_warning = False
    if hddm_w.drift_detected:
        print(f"Change detected at index {i}")
        print_warning = True
```

```text
Warning detected at index 451
Change detected at index 1077
```

## Methods

> [!NOTE]
>
> **update**
>
> Update the change detector with a single data point.
>
> **Parameters**
>
> - **x** — `bool`

## References

- Frías-Blanco I, del Campo-Ávila J, Ramos-Jimenez G, et al. Online and non-parametric drift detection methods based on Hoeffding’s bounds. IEEE Transactions on Knowledge and Data Engineering, 2014, 27(3): 810-823.
- Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer. MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.
