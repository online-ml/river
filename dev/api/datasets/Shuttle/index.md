# Shuttle

Statlog (Shuttle) dataset.

This is the UCI Statlog (Shuttle) dataset, cast as a binary anomaly-detection problem following the ODDS benchmark. The nine numerical attributes are sensor readings from a NASA space shuttle. The original seven classes are collapsed into a normal class (the majority "Rad Flow" class) and an anomaly class (the rare classes), while the second-largest class is discarded. The result is 49,097 observations of which 3,511 (\~7%) are anomalies.

The target `anomaly` is `1` for an anomaly and `0` otherwise.

## Attributes

- **desc**

  Return the description from the docstring.
- **path**

## Examples

We can compare two anomaly detectors on this dataset. The features are min-max scaled online, and performance is measured with a scale-invariant rolling ROC AUC (anomaly scores are unbounded, so a plain [`metrics.ROCAUC`](<https://riverml.xyz/dev/api/metrics/ROCAUC>) would not be appropriate).

```python
from river import anomaly
from river import datasets
from river import metrics
from river import preprocessing

dataset = datasets.Shuttle()

for name, detector in [
    ("HalfSpaceTrees", anomaly.HalfSpaceTrees(seed=42)),
    ("LODA", anomaly.LODA(seed=42)),
]:
    model = preprocessing.MinMaxScaler() | detector
    auc = metrics.RollingROCAUC(window_size=10_000)
    for x, y in dataset.take(10_000):
        auc.update(y, model.score_one(x))
        model.learn_one(x)
    print(name, auc)
```

```text
HalfSpaceTrees RollingROCAUC: 90.07%
LODA RollingROCAUC: 96.46%
```

## Methods

> [!NOTE]
>
> **take**
>
> Iterate over the k samples.
>
> **Parameters**
>
> - **k** — `int`

## References

- [UCI page](<https://archive.ics.uci.edu/dataset/148/statlog+shuttle>)
- [ODDS page](<https://odds.cs.stonybrook.edu/shuttle-dataset/>)
