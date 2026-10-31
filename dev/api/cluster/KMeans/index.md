# KMeans

Incremental k-means.

The most common way to implement batch k-means is to use Lloyd's algorithm, which consists in assigning all the data points to a set of cluster centers and then moving the centers accordingly. This requires multiple passes over the data and thus isn't applicable in a streaming setting.

In this implementation we start by finding the cluster that is closest to the current observation. We then move the cluster's central position towards the new observation. The `halflife` parameter determines by how much to move the cluster toward the new observation. You will get better results if you scale your data appropriately.

## Parameters

- **n\_clusters**

  *Default* → `5`

  Maximum number of clusters to assign.
- **halflife** *(mutable)*

  *Default* → `0.5`

  Amount by which to move the cluster centers, a reasonable value is between 0 and 1.
- **mu**

  *Default* → `0`

  Mean of the normal distribution used to instantiate cluster positions.
- **sigma**

  *Default* → `1`

  Standard deviation of the normal distribution used to instantiate cluster positions.
- **p**

  *Default* → `2`

  Power parameter for the Minkowski metric. When `p=1`, this corresponds to the Manhattan distance, while `p=2` corresponds to the Euclidean distance.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed used for generating initial centroid positions.

## Attributes

- **centers** (`dict`)

  Central positions of each cluster.

## Examples

In the following example the cluster assignments are exactly the same as when using `sklearn`'s batch implementation. However, changing the `halflife` parameter will produce different outputs.

```python
from river import cluster
from river import stream

X = [
    [1, 2],
    [1, 4],
    [1, 0],
    [-4, 2],
    [-4, 4],
    [-4, 0]
]

k_means = cluster.KMeans(n_clusters=2, halflife=0.1, sigma=3, seed=42)

for i, (x, _) in enumerate(stream.iter_array(X)):
    k_means.learn_one(x)
    print(f'{X[i]} is assigned to cluster {k_means.predict_one(x)}')
```

```text
[1, 2] is assigned to cluster 1
[1, 4] is assigned to cluster 1
[1, 0] is assigned to cluster 0
[-4, 2] is assigned to cluster 1
[-4, 4] is assigned to cluster 1
[-4, 0] is assigned to cluster 0
```

```python
k_means.predict_one({0: 0, 1: 0})
```

```text
0
```

```python
k_means.predict_one({0: 4, 1: 4})
```

```text
1
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

> [!NOTE]
>
> **learn\_predict\_one**
>
> Equivalent to `k_means.learn_one(x).predict_one(x)`, but faster.
>
> **Parameters**
>
> - **x**

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

## References

- [Sequential k-Means Clustering](<http://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm>)
- [Sculley, D., 2010, April. Web-scale k-means clustering. In Proceedings of the 19th international conference on World Wide Web (pp. 1177-1178)](<https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf>)
