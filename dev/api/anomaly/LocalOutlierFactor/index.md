# LocalOutlierFactor

Local Outlier Factor (LOF).

The LOF of a sample measures how isolated it is relative to the density of its neighbors: a value around 1 means the sample lies in a region as dense as its neighborhood, while a value substantially above 1 flags an outlier sitting in a comparatively sparse region (Breunig et al., 2000).

Samples are stored in a fixed-size sliding window managed by a nearest-neighbor `engine` (see `river.neighbors`). `learn_one` simply adds a sample to the window, so it runs in constant time and the memory footprint is bounded by the window size. `score_one` computes the LOF of a sample against the current window on demand, without modifying it. The natural streaming pattern is therefore to call `score_one` and then `learn_one` on each sample.

Because scoring is done against the window only, the result matches the static LOF computed on those samples (e.g. scikit-learn's `LocalOutlierFactor(novelty=True)` fitted on the same window).

## Parameters

- **n\_neighbors**

  *Type* → `int`

  *Default* → `10`

  The number of nearest neighbors used to define the local neighborhood.
- **engine**

  *Type* → `BaseNN | None`

  *Default* → `None`

  The nearest-neighbor search engine, which stores the samples in a sliding window and answers neighbor queries. Defaults to `neighbors.LazySearch`, an exact brute-force search over the window. Pass `neighbors.SWINN` for approximate search, or a configured engine to control the `window_size` and the distance function.

## Examples

```python
from river import anomaly

X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
lof = anomaly.LocalOutlierFactor(n_neighbors=3)

for x in X[:3]:
    lof.learn_one({"x": x})  # warming up

for x in X:
    features = {"x": x}
    print(f"Anomaly score for x={x:.3f}: {lof.score_one(features):.3f}")
    lof.learn_one(features)
```

```text
Anomaly score for x=0.500: 0.929
Anomaly score for x=0.450: 1.105
Anomaly score for x=0.430: 1.044
Anomaly score for x=0.440: 1.000
Anomaly score for x=0.445: 0.944
Anomaly score for x=0.450: 0.889
Anomaly score for x=0.000: 36.111
```

A mini-batch of samples can be learned at once from any [narwhals](<https://github.com/narwhals-dev/narwhals>)-compatible eager dataframe (pandas, polars, pyarrow, ...) with `learn_many`:

```python
import pandas as pd
from river import datasets

rows = [x for x, _ in datasets.CreditCard().take(500)]
lof = anomaly.LocalOutlierFactor(n_neighbors=20)
lof.learn_many(pd.DataFrame(rows))
[round(lof.score_one(x), 3) for x in rows[:5]]
```

```text
[1.51, 1.355, 1.987, 1.566, 1.837]
```

## Methods

> [!NOTE]
>
> **learn\_many**
>
> Update with a mini-batch of samples held in a dataframe.
>
> Any [narwhals](<https://github.com/narwhals-dev/narwhals>)-compatible eager dataframe (pandas, polars, pyarrow, ...) is accepted. Each row is added to the window in turn.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

> [!NOTE]
>
> **learn\_one**
>
> Update the model.
>
> **Parameters**
>
> - **x** — `dict`

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
> - **x** — `dict`
>
> **Returns**
>
> `float`: An anomaly score. A high score is indicative of an anomaly. A low score corresponds a

## References

- Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander (2000). LOF: Identifying Density-Based Local Outliers. In: Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. 93-104. DOI: 10.1145/342009.335388.
- David Pokrajac, Aleksandar Lazarevic, and Longin Jan Latecki (2007). Incremental Local Outlier Detection for Data Streams. In: Proceedings of the 2007 IEEE Symposium on Computational Intelligence and Data Mining (CIDM 2007). 504-515. DOI: 10.1109/CIDM.2007.368917.
