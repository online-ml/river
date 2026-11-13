# AnomalyFilter

Anomaly filter base class.

An anomaly filter has the ability to classify an anomaly score as anomalous or not. It can then be used to filter anomalies, in particular as part of a pipeline.

## Parameters

- **anomaly\_detector**

  *Type* → `AnomalyDetector`

  An anomaly detector wrapped by the anomaly filter.
- **protect\_anomaly\_detector**

  *Type* → `bool`

  *Default* → `True`

  Indicates whether or not the anomaly detector should be updated when the anomaly score is anomalous. If the data contains sporadic anomalies, then the anomaly detector should likely not be updated. Indeed, if it learns the anomaly score, then it will slowly start to consider anomalous anomaly scores as normal. This might be desirable, for instance in the case of drift.

## Methods

> [!NOTE]
>
> **classify**
>
> Classify an anomaly score as anomalous or not.
>
> **Parameters**
>
> - **score** — `float`
>
> **Returns**
>
> `bool`: A boolean value indicating whether the anomaly score is anomalous or not.

> [!NOTE]
>
> **learn\_one**
>
> Update the anomaly filter and the underlying anomaly detector.
>
> **Parameters**
>
> - **args** — `Any`
> - **learn\_kwargs** — `Any`

> [!NOTE]
>
> **score\_one**
>
> Return an outlier score.
>
> A high score is indicative of an anomaly. A low score corresponds to a normal observation.
>
> **Parameters**
>
> - **args** — `Any`
> - **kwargs** — `Any`
>
> **Returns**
>
> `float`: An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
