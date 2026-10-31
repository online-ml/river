# DDM

Drift Detection Method.

DDM (Drift Detection Method) is a concept change detection method based on the PAC learning model premise, that the learner's error rate will decrease as the number of analysed samples increase, as long as the data distribution is stationary.

If the algorithm detects an increase in the error rate, that surpasses a calculated threshold, either change is detected or the algorithm will warn the user that change may occur in the near future, which is called the warning zone.

The detection threshold is calculated in function of two statistics, obtained when \\((p\_i + s\_i)\\) is minimum:

- \\(p\_{min}\\): The minimum recorded error rate.
- \\(s\_{min}\\): The minimum recorded standard deviation.

At instant \\(i\\), the detection algorithm uses:

- \\(p\_i\\): The error rate at instant \\(i\\).
- \\(s\_i\\): The standard deviation at instant \\(i\\).

The conditions for entering the warning zone and detecting change are as follows \[see implementation note below\]:

- if \\(p\_i + s\_i \\geq p\_{min} + w\_l \* s\_{min}\\) -\> Warning zone
- if \\(p\_i + s\_i \\geq p\_{min} + d\_l \* s\_{min}\\) -\> Change detected

In the above expressions, \\(w\_l\\) and \\(d\_l\\) represent, respectively, the warning and drift thresholds.

**Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0 represents correct/normal values.

For example, if a classifier's prediction \\(y'\\) is right or wrong w.r.t. the true target label \\(y\\):

- 0: Correct, \\(y=y'\\)
- 1: Error, \\(y \\neq y'\\)

## Parameters

- **warm\_start**

  *Type* → `int`

  *Default* → `30`

  The minimum required number of analyzed samples so change can be detected. Warm start parameter for the drift detector.
- **warning\_threshold**

  *Type* → `float`

  *Default* → `2.0`

  Threshold to decide if the detector is in a warning zone. The default value gives 95\\% of confidence level to the warning assessment.
- **drift\_threshold**

  *Type* → `float`

  *Default* → `3.0`

  Threshold to decide if a drift was detected. The default value gives a 99\\% of confidence level to the drift assessment.

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
ddm = drift.binary.DDM()

data_stream = rng.choices([0, 1], k=1000)
data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])

print_warning = True
for i, x in enumerate(data_stream):
    ddm.update(x)
    if ddm.warning_detected and print_warning:
        print(f"Warning detected at index {i}")
        print_warning = False
    if ddm.drift_detected:
        print(f"Change detected at index {i}")
        print_warning = True
```

```text
Warning detected at index 1084
Change detected at index 1334
Warning detected at index 1492
```

## Methods

> [!NOTE]
>
> **update**
>
> Update the detector with a single boolean input.
>
> **Parameters**
>
> - **x** — `bool`

## References

- João Gama, Pedro Medas, Gladys Castillo, Pedro Pereira Rodrigues: Learning with Drift Detection. SBIA 2004: 286-295
