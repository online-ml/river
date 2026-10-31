# FHDDM

Fast Hoeffding Drift Detection Method.

FHDDM is a drift detection method based on the Hoeffding's inequality which uses the input average as estimator.

**Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0 represents correct/normal values.

For example, if a classifier's prediction \\(y'\\) is right or wrong w.r.t. the true target label \\(y\\):

- 0: Correct, \\(y=y'\\)
- 1: Error, \\(y \\neq y'\\)

*Implementation based on MOA.*

## Parameters

- **sliding\_window\_size**

  *Type* → `int`

  *Default* → `100`

  The minimum required number of analyzed samples so change can be detected.
- **confidence\_level**

  *Type* → `float`

  *Default* → `1e-06`

  Confidence level used to determine the epsilon coefficient in Hoeffding’s inequality. The default value gives a 99\\% of confidence level to the drift assessment.
- **short\_window\_size**

  *Type* → `int | None`

  *Default* → `None`

  The size of the short window size that it is used in a Stacking version of FHDDM.

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
fhddm = drift.binary.FHDDM()
fhddm_s = drift.binary.FHDDM(short_window_size = 20)
data_stream = rng.choices([0, 1], k=250)
data_stream = data_stream + rng.choices([0, 1], k=250, weights=[0.9, 0.1])
for i, x in enumerate(data_stream):
    fhddm.update(x)
    fhddm_s.update(x)
    if fhddm.drift_detected or fhddm_s.drift_detected:
        print(f"Change detected at index {i}")
```

```text
Change detected at index 279
Change detected at index 315
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

- A. Pesaranghader, H.L. Viktor, Fast Hoeffding Drift Detection Method for Evolving Data Streams. In the Proceedings of ECML-PKDD 2016.
- Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams.
