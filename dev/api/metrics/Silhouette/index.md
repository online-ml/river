# Silhouette

Silhouette coefficient [^/dev/api/metrics/Silhouette/#1], roughly speaking, is the ratio between cohesion and the average distances from the points to their second-closest centroid. It rewards the clustering algorithm where points are very close to their assigned centroids and far from any other centroids, that is, clustering results with good cohesion and good separation.

It rewards clusterings where points are very close to their assigned centroids and far from any other centroids, that is clusterings with good cohesion and good separation. [^/dev/api/metrics/Silhouette/#2]

The definition of Silhouette coefficient for online clustering evaluation is different from that of batch learning. It does not store information and calculate pairwise distances between all points at the same time, since the practice is too expensive for an incremental metric.

## Attributes

- **bigger\_is\_better**

  Indicates if a high value is better than a low one or not.

## Examples

```python
from river import cluster
from river import stream
from river import metrics

X = [
    [1, 2],
    [1, 4],
    [1, 0],
    [4, 2],
    [4, 4],
    [4, 0],
    [-2, 2],
    [-2, 4],
    [-2, 0]
]

k_means = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
metric = metrics.Silhouette()

for x, _ in stream.iter_array(X):
    k_means.learn_one(x)
    y_pred = k_means.predict_one(x)
    metric.update(x, y_pred, k_means.centers)

metric
```

```text
Silhouette: 0.32145
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the metric.

> [!NOTE]
>
> **revert**
>
> Revert the metric.
>
> **Parameters**
>
> - **x**
> - **y\_pred**
> - **centers**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **x**
> - **y\_pred**
> - **centers**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model** — `base.Estimator`

[^/dev/api/metrics/Silhouette/#1]: Rousseeuw, P. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis 20, 53 - 65. DOI: 10.1016/0377-0427(87)90125-7

[^/dev/api/metrics/Silhouette/#2]: Bifet, A. et al. (2018). "Machine Learning for Data Streams". DOI: 10.7551/mitpress/10654.001.0001.
