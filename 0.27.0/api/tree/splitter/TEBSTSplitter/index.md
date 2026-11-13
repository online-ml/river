# TEBSTSplitter

Truncated E-BST.

Variation of E-BST that rounds the incoming feature values before passing them to the binary search tree (BST). By doing so, the attribute observer might reduce its processing time and memory usage since small variations in the input values will end up being mapped to the same BST node.

## Parameters

- **digits**

  *Type* → `int`

  *Default* → `1`

  The number of decimal places used to round the input feature values.

## Attributes

- **is\_numeric**

  Determine whether or not the splitter works with numerical features.
- **is\_target\_class**

  Check on which kind of learning task the splitter is designed to work. If `True`, the splitter works with classification trees, otherwise it is designed for regression trees.

## Methods

> [!NOTE]
>
> **best\_evaluated\_split\_suggestion**
>
> Get the best split suggestion given a criterion and the target's statistics.
>
> **Parameters**
>
> - **criterion** — `SplitCriterion`
> - **pre\_split\_dist** — `list | dict`
> - **att\_idx** — `base.typing.FeatureName`
> - **binary\_only** — `bool` — defaults to `True`
>
> **Returns**
>
> `BranchFactory`: Suggestion of the best attribute split.

> [!NOTE]
>
> **cond\_proba**
>
> Not implemented in regression splitters.
>
> **Parameters**
>
> - **att\_val**
> - **target\_val** — `base.typing.ClfTarget`

> [!NOTE]
>
> **remove\_bad\_splits**
>
> Remove bad splits.
>
> Based on FIMT-DD's procedure to remove bad split candidates from the E-BST. This mechanism is triggered every time a split attempt fails. The rationale is to remove points whose split merit is much worse than the best candidate overall (for which the growth decision already failed). Let \\(m\_1\\) be the merit of the best split point and \\(m\_2\\) be the merit of the second best split candidate. The ratio \\(r = m\_2/m\_1\\) along with the Hoeffding bound (\\(\\epsilon\\)) are used to decide upon creating a split. A split occurs when \\(r \< 1 - \\epsilon\\). A split candidate, with merit \\(m\_i\\), is considered badr if \\(m\_i / m\_1 \< r - 2\\epsilon\\). The rationale is the following: if the merit ratio for this point is smaller than the lower bound of \\(r\\), then the true merit of that split relative to the best one is small. Hence, this candidate can be safely removed. To avoid excessive and costly manipulations of the E-BST to update the stored statistics, only the nodes whose children are all bad split points are pruned, as defined in the original paper.
>
> **Parameters**
>
> - **criterion**
> - **last\_check\_ratio** — `float`
> - **last\_check\_vr** — `float`
> - **last\_check\_e** — `float`
> - **pre\_split\_dist** — `list | dict`

> [!NOTE]
>
> **update**
>
> Update statistics of this observer given an attribute value, its target value and the weight of the instance observed.
>
> **Parameters**
>
> - **att\_val**
> - **target\_val** — `base.typing.Target`
> - **w** — `float`
