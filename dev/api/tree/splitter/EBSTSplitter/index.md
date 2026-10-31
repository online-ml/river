# EBSTSplitter

iSOUP-Tree's Extended Binary Search Tree (E-BST).

This class implements the Extended Binary Search Tree[^/dev/api/tree/splitter/EBSTSplitter/#1] (E-BST) structure, using the variant employed by Osojnik et al.[^/dev/api/tree/splitter/EBSTSplitter/#2] in the iSOUP-Tree algorithm. This structure is employed to observe the target space distribution.

Proposed along with Fast Incremental Model Tree with Drift Detection[^/dev/api/tree/splitter/EBSTSplitter/#1] (FIMT-DD), E-BST was the first attribute observer (AO) proposed for incremental Hoeffding Tree regressors. This AO works by storing all observations between splits in an extended binary search tree structure. E-BST stores the input feature realizations and statistics of the target(s) that enable calculating the split heuristic at any time. To alleviate time and memory costs, E-BST implements a memory management routine, where the worst split candidates are pruned from the binary tree.

In this variant, only the left branch statistics are stored and the complete split-enabling statistics are calculated with an in-order traversal of the binary search tree.

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

[^/dev/api/tree/splitter/EBSTSplitter/#1]: Ikonomovska, E., Gama, J., &amp; Džeroski, S. (2011). Learning model trees from evolving data streams. Data mining and knowledge discovery, 23(1), 128-168.

[^/dev/api/tree/splitter/EBSTSplitter/#2]: [Osojnik, Aljaž. 2017. Structured output prediction on Data Streams (Doctoral Dissertation)](<http://kt.ijs.si/theses/phd_aljaz_osojnik.pdf>)
