# LODA

LODA (Lightweight on-line detector of anomalies).

LODA [^/dev/api/anomaly/LODA/#1] is an ensemble of one-dimensional histograms. Each histogram approximates the probability density of the data once it has been projected onto a sparse random vector. The anomaly score of a sample is the average negative log-likelihood of its projections across the ensemble: rare projected values yield low densities and therefore high scores.

Pevný showed that aggregating many such deliberately weak detectors yields a strong anomaly detector, competitive with much heavier methods while remaining cheap to update online.

Each projection vector is sparse: only `⌊√d⌋` of the `d` features have a non-zero weight, drawn from a standard normal distribution. The feature set and the projections are fixed the first time `learn_one` is called. Features that appear later are ignored, and missing features are treated as zeros.

Unlike the histograms used in the original paper, this implementation relies on River's streaming [`sketch.Histogram`](<https://riverml.xyz/dev/api/sketch/Histogram>), which maintains a bounded number of adaptive-width bins. The density of a projected value is estimated as `(count / n) / width` of the bin that contains it, where `width` is the bin's span (or, for not-yet-merged singleton bins, the distance to the nearest neighbouring bin). Projected values that fall outside every bin are assigned a floor density, making them maximally anomalous. This keeps the detector fully online and free of any numpy dependency.

## Parameters

- **n\_bins**

  *Type* → `int`

  *Default* → `10`

  Maximum number of bins in each histogram.
- **n\_random\_cuts**

  *Type* → `int`

  *Default* → `100`

  Number of random projections (the ensemble size).
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number seed, for reproducible projections.

## Attributes

- **n\_features**

  Number of features seen during the first call to `learn_one`.

## Examples

```python
from river import anomaly
from river import datasets

loda = anomaly.LODA(n_bins=10, n_random_cuts=100, seed=42)

for x, y in datasets.CreditCard().take(2500):
    loda.learn_one(x)

loda.n_features
```

```text
30
```

```python
score = loda.score_one(x)
print(f"{score:.3f}")
```

```text
3.670
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

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
> - **x** — `dict[base.typing.FeatureName, Any]`
>
> **Returns**
>
> `float`: An anomaly score. A high score is indicative of an anomaly. A low score corresponds a

[^/dev/api/anomaly/LODA/#1]: Pevný, T., 2016. Loda: Lightweight on-line detector of anomalies. Machine Learning, 102(2), pp.275-304.
