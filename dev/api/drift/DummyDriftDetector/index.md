# DummyDriftDetector

Baseline drift detector that generates pseudo drift detection signals.

There are two approaches[^/dev/api/drift/DummyDriftDetector/#1]:

- `fixed` where the drift signal is generated every `t_0` samples.
- `random` corresponds to a pseudo-random drift detection strategy.

## Parameters

- **trigger\_method**

  *Type* → `str`

  *Default* → `fixed`

  The trigger method to use.  
  \* `fixed`  
  \* `random`
- **t\_0**

  *Type* → `int`

  *Default* → `300`

  Reference point to define triggers.
- **w**

  *Type* → `int`

  *Default* → `0`

  Auxiliary parameter whose purpose is twofold: - if `trigger_method="fixed"`, the periodic drift signals will only start after an initial warm-up period randomly defined between `\[0, w\]`. Useful to avoid that all ensemble members are reset at the same time when periodic triggers are used as the adaptation strategy. - if `trigger_method="random"`, `w` defines the probability bounds of triggering a drift. The chance of triggering a drift is \\(0.5\\) after observing `t_0` instances and becomes \\(1\\) after monitoring `t_0 + w / 2` instances. A sigmoid function is used to produce values between `\[0, 1\]` that are used as the reset probabilities.
- **dynamic\_cloning**

  *Type* → `bool`

  *Default* → `False`

  Whether to change the `seed` and `w` values each time `clone()` is called.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Attributes

- **drift\_detected**

  Whether or not a drift is detected following the last update.

## Examples

```python
import random
from river import drift

rng = random.Random(42)
```

The observed values will not affect the periodic triggers.

```python
data = [rng.gauss(0, 1) for _ in range(1000)]
```

Let's start with the fixed drift signals:

```python
ptrigger = DummyDriftDetector(t_0=500, seed=42)
for i, v in enumerate(data):
    ptrigger.update(v)
    if ptrigger.drift_detected:
        print(f"Drift detected at instance {i}.")
```

```text
Drift detected at instance 499.
Drift detected at instance 999.
```

Now, the random drift signals:

```python
rtrigger = DummyDriftDetector(
    trigger_method="random",
    t_0=500,
    w=100,
    dynamic_cloning=True,
    seed=42
)
for i, v in enumerate(data):
    rtrigger.update(v)
    if rtrigger.drift_detected:
        print(f"Drift detected at instance {i}.")
```

```text
Drift detected at instance 368.
Drift detected at instance 817.
```

Remember to set a w \> 0 value if random triggers are used:

```python
try:
    DummyDriftDetector(trigger_method="random")
except ValueError as ve:
    print(ve)
```

```text
The 'w' value must be greater than zero when 'trigger_method' is 'random'.
```

Since we set `dynamic_cloning` to `True`, a clone of the periodic trigger will have its internal parameters changed:

```python
rtrigger = rtrigger.clone()
for i, v in enumerate(data):
    rtrigger.update(v)
    if rtrigger.drift_detected:
        print(f"Drift detected at instance {i}.")
```

```text
Drift detected at instance 429.
Drift detected at instance 728.
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

## Notes

When used in ensembles, a naive implementation of periodic drift signals would make all ensemble members reset at the same time. To avoid that, the `dynamic_cloning` parameter can be set to `True`. In this case, every time the `clone` method of this detector is called in an ensemble a new `seed` is defined. If `dynamic_cloning=True` and `trigger_method="fixed"`, a new `w` between `\[0, t_0\]` will also be created for the new cloned instance.

[^/dev/api/drift/DummyDriftDetector/#1]: Heitor Gomes, Jacob Montiel, Saulo Martiello Mastelini, Bernhard Pfahringer, and Albert Bifet. On Ensemble Techniques for Data Stream Regression. IJCNN'20. International Joint Conference on Neural Networks. 2020.
