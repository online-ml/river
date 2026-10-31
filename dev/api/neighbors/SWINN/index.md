# SWINN

Sliding WIndow-based Nearest Neighbor (SWINN) search using Graphs.

Extends the NNDescent algorithm[^/dev/api/neighbors/SWINN/#1] to handle vertex addition and removal in a FIFO data ingestion policy. SWINN builds and keeps a directed graph where edges connect the nearest neighbors. Any distance metric can be used to build the graph. By using a directed graph, the user must set the desired number of neighbors. More neighbors imply more accurate search queries at the cost of increased running time and memory usage. Note that although the number of directed neighbors is limited by the user, there is no direct control on the number of reverse neighbors, i.e., the number of vertices that have an edge to a given vertex.

The basic idea of SWINN and NNDescent is that "the neighbor of my neighbors might as well be my neighbor". Hence, the connections are constantly revisited to improve the graph structure. The algorithm for creating and maintaining the search graph can be described in general lines as follows:

- Start with a random neighborhood graph;
- For each node in the search graph: refine the current neighborhood by checking if there are better neighborhood options among the neighbors of the current neighbors;
- If the total number of neighborhood changes is smaller than a given stopping criterion, then stop.

SWINN adds strategies to remove vertices from the search graph and pruning redundant edges. SWINN is more efficient when the selected `maxlen` is greater than 500. For small sized data windows, using the lazy/exhaustive search, i.e., [`neighbors.LazySearch`](<https://riverml.xyz/dev/api/neighbors/LazySearch>) might be a better idea.

## Parameters

- **graph\_k**

  *Type* → `int`

  *Default* → `20`

  The maximum number of direct nearest neighbors each node has.
- **dist\_func**

  *Type* → `DistanceFunc | FunctionWrapper | None`

  *Default* → `None`

  The distance function used to compare two items. If not set, use the Minkowski distance with `p=2`.
- **maxlen**

  *Type* → `int`

  *Default* → `1000`

  The maximum size of the data buffer.
- **warm\_up**

  *Type* → `int`

  *Default* → `500`

  How many data instances to observe before starting the search graph.
- **max\_candidates**

  *Type* → `int | None`

  *Default* → `None`

  The maximum number of vertices to consider when performing local neighborhood joins. If not set SWINN will use `min(50, max(50, self.graph_k))`.
- **delta**

  *Type* → `float`

  *Default* → `0.0001`

  Early stop parameter for the neighborhood refinement procedure. NNDescent will stop running if the maximum number of iterations is reached or the number of edge changes after an iteration is smaller than or equal to `delta * graph_k * n_nodes`. In the last expression, `n_nodes` refers to the number of graph nodes involved in the (local) neighborhood refinement.
- **prune\_prob**

  *Type* → `float`

  *Default* → `0.0`

  The probability of removing redundant edges. Must be between `0` and `1`. If set to zero, no edge will be pruned. When set to one, every potentially redundant edge will be dropped.
- **n\_iters**

  *Type* → `int`

  *Default* → `10`

  The maximum number of NNDescent iterations to perform to refine the search index.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Methods

> [!NOTE]
>
> **append**
>
> Add a new item to the search index.
>
> Data is stored using the FIFO strategy. Both the data buffer and the search graph are updated. The addition of a new item will trigger the removal of the oldest item, if the maximum size was reached. All edges of the removed node are also dropped and safety procedures are applied to ensure its neighbors keep accessible. The addition of a new item also trigger local neighborhood refinement procedures, to ensure the search index is effective and the node degree constraints are met.
>
> **Parameters**
>
> - **item** — `typing.Any`
> - **kwargs** — `typing.Any`

> [!NOTE]
>
> **connectivity**
>
> Get a list with the size of each connected component in the search graph.
>
> This metric provides an overview of reachability in the search index by using Kruskal's algorithm to build a forest of connected components. We want our search index to have a single connected component, i.e., the case where we get a list containing a single number which is equal to `maxlen`. If that is not the case, not every node in the search graph can be reached from any given starting point. You may want to try increasing `graph_k` to improve connectivity. However, keep in mind the following aspects: 1) computing this metric is a costly operation (\\(O(E\\log V)\\)), where \\(E\\) and \\(V\\) are, respectively, the number of edges and vertices in the search graph; 2) often, connectivity comes at the price of increased computational costs. Tweaking the `sample_rate` might help in such situations. The best possible scenario is to decrease the value of `graph_k` while keeping a single connected component.
>
> **Returns**
>
> `list&#91;int&#93;`: A list of the number of elements in each connected component of the graph.

> [!NOTE]
>
> **refresh\_targets**
>
> Refresh the set of classes in the window. Used by classifiers where labels are added as \[1\] in the vertex tuple.
>
> This is used to clean up classes that are no longer in the window, and ensure we do not consider "None" a class. It is called every `cleanup_every` step, or can be called manually.

> [!NOTE]
>
> **search**
>
> Search the underlying nearest neighbor graph given a query item.
>
> In case not enough samples were observed, i.e., the number of stored samples is smaller than `warm_up`, then the search switches to a brute force strategy.
>
> **Parameters**
>
> - **item** — `typing.Any`
> - **n\_neighbors** — `int`
> - **epsilon** — `float` — defaults to `0.1`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `tuple&#91;list&#91;typing.Any&#93;, list&#91;float&#93;&#93;`: neighbors, dists

## Notes

There is an accuracy/speed trade-off between `graph_k` and `sample_rate`. To ensure a single connected component, and thus an effective search index, one can increase `graph_k`. The `connectivity` method is a helper to determine whether the search index has a single connected component. However, search accuracy might come at the cost of increased memory usage and slow processing. To alleviate that, one can rely on decreasing the `sample_rate` to avoid exploring all the undirected edges of a node during search queries and local graph refinements. Moreover, the edge pruning procedures also help decreasing the computational costs. Note that, anything that limits the number of explored neighbors or prunes edges might have a negative impact on search accuracy.

[^/dev/api/neighbors/SWINN/#1]: Dong, W., Moses, C., &amp; Li, K. (2011, March). Efficient k-nearest neighbor graph construction for generic similarity measures. In Proceedings of the 20th international conference on World wide web (pp. 577-586).
