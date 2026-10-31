# ADWIN

Adaptive Windowing method for concept drift detection[^/dev/api/drift/ADWIN/#1].

ADWIN (ADaptive WINdowing) is a popular drift detection method with mathematical guarantees. ADWIN efficiently keeps a variable-length window of recent items; such that it holds that there has no been change in the data distribution. This window is further divided into two sub-windows \\((W\_0, W\_1)\\) used to determine if a change has happened. ADWIN compares the average of \\(W\_0\\) and \\(W\_1\\) to confirm that they correspond to the same distribution. Concept drift is detected if the distribution equality no longer holds. Upon detecting a drift, \\(W\_0\\) is replaced by \\(W\_1\\) and a new \\(W\_1\\) is initialized. ADWIN uses a significance value \\(\\delta=\\in(0,1)\\) to determine if the two sub-windows correspond to the same distribution.

## Parameters

- **delta**

  *Type* → `float`

  *Default* → `0.002`

  Significance value.
- **clock**

  *Type* → `int`

  *Default* → `32`

  How often ADWIN should check for changes. 1 means every new data point, default is 32. Higher values speed up processing, but may also lead to increased delay in change detection.
- **max\_buckets**

  *Type* → `int`

  *Default* → `5`

  The maximum number of buckets of each size that ADWIN should keep before merging buckets. The idea of data buckets comes from the compression algorithm introduced in the ADWIN2, the second iteration of the ADWIN algorithm presented in the original research paper. This is the ADWIN version available in River.
- **min\_window\_length**

  *Type* → `int`

  *Default* → `5`

  The minimum length allowed for a subwindow when checking for concept drift. Subwindows whose size is smaller than this value will be ignored during concept drift evaluation. Lower values may decrease delay in change detection but may also lead to more false positives.
- **grace\_period**

  *Type* → `int`

  *Default* → `10`

  ADWIN does not perform any change detection until at least this many data points have arrived.

## Attributes

- **drift\_detected**

  Whether or not a drift is detected following the last update.
- **estimation**

  Estimate of mean value in the window.
- **n\_detections**

  The total number of detected changes.
- **total**

  The sum of the stored elements.
- **variance**

  The sample variance within the stored (adaptive) window.
- **width**

  Window size

## Examples

```python
import random
from river import drift

rng = random.Random(12345)
adwin = drift.ADWIN()

data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)

for i, val in enumerate(data_stream):
    adwin.update(val)
    if adwin.drift_detected:
        print(f"Change detected at index {i}, input value: {val}")
```

```text
Change detected at index 1023, input value: 4
```

## Methods

> [!NOTE]
>
> **update**
>
> Update the change detector with a single data point.
>
> Apart from adding the element value to the window, by inserting it in the correct bucket, it will also update the relevant statistics, in this case the total sum of all values, the window width and the total variance.
>
> **Parameters**
>
> - **x** — `int | float`

[^/dev/api/drift/ADWIN/#1]: Albert Bifet and Ricard Gavalda. "Learning from time-changing data with adaptive windowing." In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448. Society for Industrial and Applied Mathematics, 2007.
