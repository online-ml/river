# ODAC

The Online Divisive-Agglomerative Clustering (ODAC)[^/dev/api/cluster/ODAC/#1] aims at continuously maintaining a hierarchical cluster structure from evolving time series data streams.

The distance between time-series a and b is given by `rnomc(a, b) = sqrt((1 - corr(a, b)) / 2)`, where `corr(a, b)` is the Pearson Correlation coefficient. If the cluster has only one time-series, the diameter is given by the time-series variance. The cluster's diameter is given by the largest distance between the cluster's time-series.

ODAC continuously monitors the evolution of diameters, only of the leaves, and splits or merges them by gathering more data or reacting to concept drift - a confidence level from the Hoeffding bound supports such changes.

So, the split operator, where the Hoeffding bound is applied, occurs when the difference between the largest distance (diameter) and the second largest difference is greater than a constant. Furthermore, the merge operator checks if one of the cluster's children has a diameter bigger than their parent - applying the Hoeffding bound again.

## Parameters

- **confidence\_level**

  *Type* → `float`

  *Default* → `0.9`

  The confidence level that user wants to work.
- **n\_min**

  *Type* → `int`

  *Default* → `100`

  Number of minimum observations to gather before checking whether or not clusters must be split or merged.
- **tau**

  *Type* → `float`

  *Default* → `0.1`

  Threshold below which a split will be forced to break ties.

## Attributes

- **structure\_changed** (`bool`)

  This variable is true when the structure changed, produced by splitting or aggregation.

## Examples

```python
from river import cluster
from river.datasets import synth

model = cluster.ODAC()

dataset = synth.FriedmanDrift(drift_type='gra', position=(150, 200), seed=42)

for i, (x, _) in enumerate(dataset.take(500)):
    model.learn_one(x)
    if model.structure_changed:
        print(f"Structure changed at observation {i + 1}")
```

```text
Structure changed at observation 1
Structure changed at observation 100
Structure changed at observation 200
Structure changed at observation 300
```

```python
print(model.render_ascii())
```

```text
ROOT d1=0.79 d2=0.76 [NOT ACTIVE]
├── CH1_LVL_1 d1=0.74 d2=0.72 [NOT ACTIVE]
│   ├── CH1_LVL_2 d1=0.08 [3]
│   └── CH2_LVL_2 d1=0.73 [2, 4]
└── CH2_LVL_1 d1=0.81 d2=0.78 [NOT ACTIVE]
    ├── CH1_LVL_2 d1=0.73 d2=0.67 [NOT ACTIVE]
    │   ├── CH1_LVL_3 d1=0.72 [0, 9]
    │   └── CH2_LVL_3 d1=0.08 [1]
    └── CH2_LVL_2 d1=0.74 d2=0.73 [NOT ACTIVE]
        ├── CH1_LVL_3 d1=0.71 [5, 6]
        └── CH2_LVL_3 d1=0.71 [7, 8]
```

You can access some properties of the clustering model directly:

```python
model.n_clusters
```

```text
11
```

```python
model.n_active_clusters
```

```text
6
```

```python
model.height
```

```text
3
```

These properties are also available in a summarized form:

```python
model.summary
```

```text
{'n_clusters': 11, 'n_active_clusters': 6, 'height': 3}
```

## Methods

> [!NOTE]
>
> **draw**
>
> Method to draw the hierarchical cluster's structure as a Graphviz graph.
>
> **Parameters**
>
> - **max\_depth** — `int | None` — defaults to `None`
> - **show\_clusters\_info** — `list[typing.Hashable]` — defaults to `['timeseries_names', 'd1', 'd2', 'e']`
> - **n\_decimal\_places** — `int` — defaults to `2`

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict`

> [!NOTE]
>
> **predict\_one**
>
> This algorithm does not predict anything. It builds a hierarchical cluster's structure.
>
> **Parameters**
>
> - **x** — `dict`

> [!NOTE]
>
> **render\_ascii**
>
> Method to render the hierarchical cluster's structure in text format.
>
> **Parameters**
>
> - **n\_decimal\_places** — `int` — defaults to `2`

[^/dev/api/cluster/ODAC/#1]: P. P. Rodrigues, J. Gama and J. Pedroso, "Hierarchical Clustering of Time-Series Data Streams" in IEEE Transactions on Knowledge and Data Engineering, vol. 20, no. 5, pp. 615-627, May 2008, doi: 10.1109/TKDE.2007.190727.
