# EDDM

Early Drift Detection Method.

EDDM (Early Drift Detection Method) aims to improve the detection rate of gradual concept drift in DDM, while keeping a good performance against abrupt concept drift.

This method works by keeping track of the average distance between two errors instead of only the error rate. For this, it is necessary to keep track of the running average distance and the running standard deviation, as well as the maximum distance and the maximum standard deviation.

The algorithm works similarly to the DDM algorithm, by keeping track of statistics only. It works with the running average distance (\\(p\_i'\\)) and the running standard deviation (\\(s\_i'\\)), as well as \\(p'\_{max}\\) and \\(s'\_{max}\\), which are the values of \\(p\_i'\\) and \\(s\_i'\\) when \\((p\_i' + 2 \* s\_i')\\) reaches its maximum.

Like DDM, there are two threshold values that define the borderline between no change, warning zone, and drift detected. These are as follows:

- if \\((p\_i' + 2 \* s\_i') / (p'\_{max} + 2 \* s'\_{max}) \< \\alpha\\) -\> Warning zone
- if \\((p\_i' + 2 \* s\_i') / (p'\_{max} + 2 \* s'\_{max}) \< \\beta\\) -\> Change detected

\\(\\alpha\\) and \\(\\beta\\) are set to 0.95 and 0.9, respectively.

**Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0 represents correct/normal values.

For example, if a classifier's prediction \\(y'\\) is right or wrong w.r.t. the true target label \\(y\\):

- 0: Correct, \\(y=y'\\)
- 1: Error, \\(y \\\\neq y'\\)

## Parameters

- **warm\_start**

  *Type* → `int`

  *Default* → `30`

  The minimum required number of monitored errors/failures so change can be detected. Warm start parameter for the drift detector.
- **alpha**

  *Type* → `float`

  *Default* → `0.95`

  Threshold for triggering a warning. Must be between 0 and 1. The smaller the value, the more conservative the detector becomes.
- **beta**

  *Type* → `float`

  *Default* → `0.9`

  Threshold for triggering a drift. Must be between 0 and 1. The smaller the value, the more conservative the detector becomes.

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
eddm = drift.binary.EDDM(alpha=0.8, beta=0.75)

data_stream = rng.choices([0, 1], k=1000)
data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])

print_warning = True
for i, x in enumerate(data_stream):
    eddm.update(x)
    if eddm.warning_detected and print_warning:
        print(f"Warning detected at index {i}")
        print_warning = False
    if eddm.drift_detected:
        print(f"Change detected at index {i}")
        print_warning = True
```

```text
Warning detected at index 1059
Change detected at index 1278
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

- Early Drift Detection Method. Manuel Baena-Garcia, Jose Del Campo-Avila, Raúl Fidalgo, Albert Bifet, Ricard Gavalda, Rafael Morales-Bueno. In Fourth International Workshop on Knowledge Discovery from Data Streams, 2006.
