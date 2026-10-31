# ExhaustiveSplitter

Numeric attribute observer for classification tasks that is based on a Binary Search Tree.

This algorithm[^/dev/api/tree/splitter/ExhaustiveSplitter/#1] is also referred to as exhaustive attribute observer, since it ends up storing all the observations between split attempts[^/dev/api/tree/splitter/ExhaustiveSplitter/#2].

This splitter cannot perform probability density estimations, so it does not work well when coupled with tree leaves using naive bayes models.

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
> - **binary\_only** — `bool`
>
> **Returns**
>
> `BranchFactory`: Suggestion of the best attribute split.

> [!NOTE]
>
> **cond\_proba**
>
> The underlying data structure used to monitor the input does not allow probability density estimations. Hence, it always returns zero for any given input.
>
> **Parameters**
>
> - **att\_val**
> - **target\_val** — `base.typing.ClfTarget`

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

[^/dev/api/tree/splitter/ExhaustiveSplitter/#1]: Domingos, P. and Hulten, G., 2000, August. Mining high-speed data streams. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 71-80).

[^/dev/api/tree/splitter/ExhaustiveSplitter/#2]: Pfahringer, B., Holmes, G. and Kirkby, R., 2008, May. Handling numeric attributes in hoeffding trees. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 296-307). Springer, Berlin, Heidelberg.
