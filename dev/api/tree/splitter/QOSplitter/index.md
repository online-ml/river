# QOSplitter

Quantization observer (QO).

This splitter utilizes a hash-based quantization algorithm to keep track of the target statistics and evaluate split candidates. QO, relies on the radius parameter to define discretization intervals for each incoming feature. Split candidates are defined as the midpoints between two consecutive hash slots. Both binary splits and multi-way splits can be created by this attribute observer. This class implements the algorithm described in [^/dev/api/tree/splitter/QOSplitter/#1].

The smaller the quantization radius, the more hash slots will be created to accommodate the discretized data. Hence, both the running time and memory consumption increase, but the resulting splits ought to be closer to the ones obtained by a batch exhaustive approach. On the other hand, if the radius is too large, fewer slots will be created, less memory and running time will be required, but at the cost of coarse split suggestions.

QO assumes that all features have the same range. It is always advised to scale the features to apply this splitter. That can be done using the `preprocessing` module. A good "rule of thumb" is to scale data using [`preprocessing.StandardScaler`](<https://riverml.xyz/dev/api/preprocessing/StandardScaler>) and define the radius as a proportion of the features' standard deviation. For instance, the default radius value would correspond to one quarter of the normalized features' standard deviation (since the scaled data has zero mean and unit variance). If the features come from normal distributions, by following the empirical rule, roughly `32` hash slots will be created.

## Parameters

- **radius**

  *Type* → `float`

  *Default* → `0.25`

  The quantization radius. QO discretizes the incoming feature in intervals of equal length that are defined by this parameter.
- **allow\_multiway\_splits**

  *Default* → `False`

  Whether or not allow that multiway splits are evaluated. Numeric multi-way splits use the same quantization strategy of QO to create multiple tree branches. The same quantization radius is used, and each stored slot represents the split enabling statistics of one branch.

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

[^/dev/api/tree/splitter/QOSplitter/#1]: Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization to perform split attempts in online tree regressors. Pattern Recognition Letters.
