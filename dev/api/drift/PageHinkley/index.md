# PageHinkley

Page-Hinkley method for concept drift detection.

This change detection method works by computing the observed values and their mean up to the current moment. Page-Hinkley does not signal warning zones, only change detections.

This detector implements the CUSUM control chart for detecting changes. This implementation also supports the two-sided Page-Hinkley test to detect increasing and decreasing changes in the mean of the input values.

## Parameters

- **min\_instances**

  *Type* → `int`

  *Default* → `30`

  The minimum number of instances before detecting change.
- **delta**

  *Type* → `float`

  *Default* → `0.005`

  The delta factor for the Page-Hinkley test.
- **threshold**

  *Type* → `float`

  *Default* → `50.0`

  The change detection threshold (lambda).
- **alpha**

  *Type* → `float`

  *Default* → `0.9999`

  The forgetting factor, used to weight the observed value and the mean.
- **mode**

  *Type* → `str`

  *Default* → `both`

  Whether to consider increases ("up"), decreases ("down") or both ("both") when monitoring the fading mean.

## Attributes

- **drift\_detected**

  Whether or not a drift is detected following the last update.

## Examples

```python
import random
from river import drift

rng = random.Random(12345)
ph = drift.PageHinkley()

data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

for i, val in enumerate(data_stream):
    ph.update(val)
    if ph.drift_detected:
        print(f"Change detected at index {i}, input value: {val}")
```

```text
Change detected at index 1006, input value: 5
```

## Methods

> [!NOTE]
>
> **update**
>
> Update the detector with a single data point.
>
> **Parameters**
>
> - **x** — `int | float`

## References

- E. S. Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2 (1954), 100-115.
- Sebastião, R., &amp; Fernandes, J. M. (2017, June). Supporting the Page-Hinkley test with empirical mode decomposition for change detection. In International Symposium on Methodologies for Intelligent Systems (pp. 492-498). Springer, Cham.
