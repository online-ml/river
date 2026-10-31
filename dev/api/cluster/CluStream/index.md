# CluStream

CluStream

The CluStream algorithm [^/dev/api/cluster/CluStream/#1] maintains statistical information about the data using micro-clusters. These micro-clusters are temporal extensions of cluster feature vectors. The micro-clusters are stored at snapshots in time following a pyramidal pattern. This pattern allows to recall summary statistics from different time horizons.

Training with a new point `p` is performed in two main tasks:

- Determinate the closest micro-cluster to `p`.
- Check whether `p` fits (memory) into the closest micro-cluster:

  - if `p` fits, put into micro-cluster
  - if `p` does not fit, free some space to insert a new micro-cluster.

  This is done in two ways, delete an old micro-cluster or merge the two micro-clusters closest to each other.

This implementation is an improved version from the original algorithm. Instead of calculating the traditional cluster feature vector of the number of observations, linear sum and sum of squares of data points and time stamps, this implementation adopts the use of Welford's algorithm [^/dev/api/cluster/CluStream/#2] to calculate the incremental variance, facilitated through [`stats.Var`](<https://riverml.xyz/dev/api/stats/Var>) available within River.

Since River does not support an actual "off-line" phase of the clustering algorithm (as data points are assumed to arrive continuously, one at a time), a `time_gap` parameter is introduced. After each `time_gap`, an incremental K-Means clustering algorithm will be initialized and applied on currently available micro-clusters to form the final solution, i.e. macro-clusters.

## Parameters

- **n\_macro\_clusters**

  *Type* → `int`

  *Default* → `5`

  The number of clusters (k) for the k-means algorithm.
- **max\_micro\_clusters**

  *Type* → `int`

  *Default* → `100`

  The maximum number of micro-clusters to use.
- **micro\_cluster\_r\_factor**

  *Type* → `int`

  *Default* → `2`

  Multiplier for the micro-cluster radius. When deciding to add a new data point to a micro-cluster, the maximum boundary is defined as a factor of the `micro_cluster_r_factor` of the RMS deviation of the data points in the micro-cluster from the centroid.
- **time\_window**

  *Type* → `int`

  *Default* → `1000`

  If the current time is `T` and the time window is `h`, we only consider the data that arrived within the period `(T-h,T)`.
- **time\_gap**

  *Type* → `int`

  *Default* → `100`

  An incremental k-means is applied on the current set of micro-clusters after each `time_gap` to form the final macro-cluster solution.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed used for generating initial centroid positions.

## Attributes

- **centers** (`dict`)

  Central positions of each cluster.

## Examples

In the following example, `max_micro_clusters` is set relatively low due to the limited number of training points. Moreover, all points are learnt before any predictions are made. The `halflife` is set at 0.4, to show that you can pass [`cluster.KMeans`](<https://riverml.xyz/dev/api/cluster/KMeans>) parameters via keyword arguments.

```python
from river import cluster
from river import stream

X = [
    [1, 2],
    [1, 4],
    [1, 0],
    [-4, 2],
    [-4, 4],
    [-4, 0],
    [5, 0],
    [5, 2],
    [5, 4]
]

clustream = cluster.CluStream(
    n_macro_clusters=3,
    max_micro_clusters=5,
    time_gap=3,
    seed=0,
    halflife=0.4
)

for x, _ in stream.iter_array(X):
    clustream.learn_one(x)

clustream.predict_one({0: 1, 1: 1})
```

```text
1
```

```python
clustream.predict_one({0: -4, 1: 3})
```

```text
2
```

```python
clustream.predict_one({0: 4, 1: 3.5})
```

```text
0
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[typing.FeatureName, Any]`
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **predict\_one**
>
> Predicts the cluster number for a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[typing.FeatureName, Any]`
>
> **Returns**
>
> `int`: A cluster number.

[^/dev/api/cluster/CluStream/#1]: Aggarwal, C.C., Philip, S.Y., Han, J. and Wang, J., 2003, A framework for clustering evolving data streams. In Proceedings 2003 VLDB conference (pp. 81-92). Morgan Kaufmann.

[^/dev/api/cluster/CluStream/#2]: Chan, T.F., Golub, G.H. and LeVeque, R.J., 1982. Updating formulae and a pairwise algorithm for computing sample variances. In COMPSTAT 1982 5th Symposium held at Toulouse 1982 (pp. 30-41). Physica, Heidelberg. https://doi.org/10.1007/978-3-642-51461-6\_3.
