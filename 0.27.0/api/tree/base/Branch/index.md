# Branch

A generic tree branch.

## Attributes

- **height**

  Distance to the deepest descendant.
- **n\_branches**

  Number of branches, including thyself.
- **n\_leaves**

  Number of leaves.
- **n\_nodes**

  Number of descendants, including thyself.
- **repr\_split**

  String representation of the split.

## Methods

> [!NOTE]
>
> **iter\_bfs**
>
> Iterate over nodes in breadth-first order.

> [!NOTE]
>
> **iter\_branches**
>
> Iterate over branches in depth-first order.

> [!NOTE]
>
> **iter\_dfs**
>
> Iterate over nodes in depth-first order.

> [!NOTE]
>
> **iter\_edges**
>
> Iterate over edges in depth-first order.

> [!NOTE]
>
> **iter\_leaves**
>
> Iterate over leaves from the left-most one to the right-most one.

> [!NOTE]
>
> **most\_common\_path**
>
> Return a tuple with the branch index and the child node related to the most traversed path.
>
> Used in case the split feature is missing from an instance.

> [!NOTE]
>
> **next**
>
> Move to the next node down the tree.
>
> **Parameters**
>
> - **x**

> [!NOTE]
>
> **to\_dataframe**
>
> Build a DataFrame containing one record for each node.

> [!NOTE]
>
> **traverse**
>
> Return the leaf corresponding to the given input.
>
> **Parameters**
>
> - **x**
> - **until\_leaf** — defaults to `True`

> [!NOTE]
>
> **walk**
>
> Iterate over the nodes of the path induced by x.
>
> **Parameters**
>
> - **x**
> - **until\_leaf** — defaults to `True`
