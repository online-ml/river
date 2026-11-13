# NoDrift

Dummy class used to turn off concept drift detection capabilities of adaptive models.

It always signals that no concept drift was detected.

## Attributes

- **drift\_detected**

  Whether or not a drift is detected following the last update.

## Examples

```python
from river import drift
from river import evaluate
from river import forest
from river import metrics
from river.datasets import synth

dataset = synth.ConceptDriftStream(
    seed=8,
    position=500,
    width=40,
).take(700)
```

We can turn off the warning detection capabilities of Adaptive Random Forest (ARF) or other similar models. Thus, the base models will reset immediately after identifying a drift, bypassing the background model building phase:

```python
adaptive_model = forest.ARFClassifier(
    leaf_prediction="mc",
    warning_detector=drift.NoDrift(),
    seed=8
)
```

We can also turn off the concept drift handling capabilities completely:

```python
stationary_model = forest.ARFClassifier(
    leaf_prediction="mc",
    warning_detector=drift.NoDrift(),
    drift_detector=drift.NoDrift(),
    seed=8
)
```

Let's put that to test:

```python
for x, y in dataset:
    adaptive_model.learn_one(x, y)
    stationary_model.learn_one(x, y)
```

The adaptive model:

```python
adaptive_model.n_drifts_detected()
```

```text
2
```

```python
adaptive_model.n_warnings_detected()
```

```text
0
```

The stationary one:

```python
stationary_model.n_drifts_detected()
```

```text
0
```

```python
stationary_model.n_warnings_detected()
```

```text
0
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
