# LazySearch

Exact nearest neighbors using a lazy search estrategy.

## Parameters

- **window\_size**

  *Type* → `int`

  *Default* → `50`

  Size of the sliding window use to search neighbors with.
- **min\_distance\_keep**

  *Type* → `float`

  *Default* → `0.0`

  The minimum distance (similarity) to consider adding a point to the window. E.g., a value of 0.0 will add even exact duplicates.
- **dist\_func**

  *Type* → `DistanceFunc | FunctionWrapper | None`

  *Default* → `None`

  A distance function which accepts two input items to compare. If not set, use the Minkowski distance with `p=2`.

## Methods

> [!NOTE]
>
> **append**
>
> Add a point to the window, optionally with extra metadata.
>
> **Parameters**
>
> - **item** — `typing.Any`
> - **extra** — `typing.Any | None` — defaults to `None`
> - **kwargs** — `typing.Any`

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
> Find the `n_neighbors` closest points to `item`, along with their distances.
>
> **Parameters**
>
> - **item** — `typing.Any`
> - **n\_neighbors** — `int`
> - **kwargs** — `typing.Any`

> [!NOTE]
>
> **update**
>
> Update the window with a new point, only added if \> min distance.
>
> If min distance is 0, we do not need to do the calculation. The item (and extra metadata) will not be added to the window if it is too close to an existing point.
>
> **Parameters**
>
> - **item** — `typing.Any`
> - **n\_neighbors** — `int` — defaults to `1`
> - **extra** — `typing.Any | None` — defaults to `None`
>
> **Returns**
>
> `bool`: A boolean (true/false) to indicate if the point was added.

## Notes

Updates are by default stored by the FIFO (first in first out) method, which means that when the size limit is reached, old samples are dumped to give room for new samples. This is circular, meaning that older points are dumped first. This also gives the implementation a temporal aspect, because older samples are replaced with newer ones.

The parameter `min_dinstance_keep` controls the addition of new items to the window - items that are far enough away (\> min\_distance\_keep) are added to the window. Thus a value of 0 indicates that we add all points, and increasing from 0 makes it less likely we will keep a new item.
