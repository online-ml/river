# GaussianSplitter

Numeric attribute observer for classification tasks that is based on Gaussian estimators.

The distribution of each class is approximated using a Gaussian distribution. Hence, the probability density function can be easily calculated.

## Parameters

- **n\_splits**

  *Type* → `int`

  *Default* → `10`

  The number of partitions to consider when querying for split candidates.

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
> **cond\_log\_proba**

> [!NOTE]
>
> **cond\_proba**
>
> Get the probability for an attribute value given a class.
>
> **Parameters**
>
> - **att\_val**
> - **target\_val** — `base.typing.ClfTarget`
>
> **Returns**
>
> `float`: Probability for an attribute value given a class.

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
