# Clustering

Online clustering groups data points into clusters one sample at a time, without storing the full dataset. This recipe shows how to set up a clustering pipeline and evaluate it with River's built-in tools.

## A basic clustering loop

The core interface mirrors supervised learning — `learn_one` ingests a sample, `predict_one` returns a cluster label:

```python
from river import cluster, stream

X = [
    [1, 2], [1, 4], [1, 0],
    [4, 2], [4, 4], [4, 0],
    [-2, 2], [-2, 4], [-2, 0],
]

model = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)

for x, _ in stream.iter_array(X):
    model.learn_one(x)
    label = model.predict_one(x)
    print(f"{x} → cluster {label}")
```

```text
{0: 1, 1: 2} → cluster 1
{0: 1, 1: 4} → cluster 1
{0: 1, 1: 0} → cluster 1
{0: 4, 1: 2} → cluster 1
{0: 4, 1: 4} → cluster 1
{0: 4, 1: 0} → cluster 1
{0: -2, 1: 2} → cluster 2
{0: -2, 1: 4} → cluster 2
{0: -2, 1: 0} → cluster 2
```

Because clustering is unsupervised, the `y` in each `(x, y)` pair is ignored — `learn_one` only takes `x`.

## Evaluation with the Silhouette coefficient

River provides internal clustering metrics that don't need ground-truth labels. The `Silhouette` metric measures how close points are to their assigned centroid compared to the next-closest centroid (lower is better):

```python
from river import metrics

model = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
metric = metrics.Silhouette()

for x, _ in stream.iter_array(X):
    y_pred = model.predict_one(x)
    metric.update(x, y_pred, model.centers)
    model.learn_one(x)

metric
```

```text
Silhouette: 0.53575
```

Note that clustering metrics require three arguments: the features `x`, the predicted cluster `y_pred`, and the current cluster `centers`.

## Using `progressive_val_score`

You can also use [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>) with a clustering metric. It handles the predict/update/learn cycle for you, including passing the features and cluster centers to the metric automatically:

```python
from river import evaluate

model = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)

evaluate.progressive_val_score(
    dataset=stream.iter_array(X),
    model=model,
    metric=metrics.Silhouette(),
)
```

```text
Silhouette: 0.53575
```

## Debugging a pipeline

`debug_one` shows the state of a sample as it passes through each pipeline step. Keep in mind that `debug_one` only calls `transform_one` — it does **not** train the pipeline. If the scaler hasn't seen any data yet, all values will appear as zero. Train the model first, then use `debug_one` to inspect:

```python
from river import compose, preprocessing

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0),
)

# Train on some data first
for x, _ in stream.iter_array(X):
    model.learn_one(x)

# Now debug_one shows meaningful scaled values
sample = {0: 1.0, 1: 2.0}
print(model.debug_one(sample))
```

```text
0. Input
--------
0: 1.00000 (float)
1: 2.00000 (float)

1. StandardScaler
-----------------
0: 0.00000 (float)
1: 0.00000 (float)

2. KMeans
---------
Prediction: 1
```
