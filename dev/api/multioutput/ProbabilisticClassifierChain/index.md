# ProbabilisticClassifierChain

Probabilistic Classifier Chains.

The Probabilistic Classifier Chains (PCC) [^/dev/api/multioutput/ProbabilisticClassifierChain/#1] is a Bayes-optimal method based on the Classifier Chains (CC).

Consider the concept of chaining classifiers as searching a path in a binary tree whose leaf nodes are associated with a label \\(y \\in Y\\). While CC searches only a single path in the aforementioned binary tree, PCC looks at each of the \\(2^l\\) paths, where \\(l\\) is the number of labels. This limits the applicability of the method to data sets with a small to moderate number of labels. The authors recommend no more than about 15 labels for real-world applications.

## Parameters

- **model**

  *Type* → `base.Classifier`

## Examples

```python
from river import linear_model
from river import metrics
from river import multioutput
from river.datasets import synth

dataset = synth.Logical(seed=42, n_tiles=100)

model = multioutput.ProbabilisticClassifierChain(
    model=linear_model.LogisticRegression()
)

metric = metrics.multioutput.MicroAverage(metrics.Jaccard())

for x, y in dataset:
   y_pred = model.predict_one(x)
   y_pred = {k: y_pred.get(k, 0) for k in y}
   metric.update(y, y_pred)
   model.learn_one(x, y)

metric
```

```text
MicroAverage(Jaccard): 51.84%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and the labels `y`.
>
> **Parameters**
>
> - **x**
> - **y**
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the labels of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[FeatureName, typing.Any]`
> - **kwargs** — `typing.Any`
>
> **Returns**
>
> `dict&#91;FeatureName, bool&#93;`: The predicted labels.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label appearing given dictionary of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.

[^/dev/api/multioutput/ProbabilisticClassifierChain/#1]: Cheng, W., Hüllermeier, E., &amp; Dembczynski, K. J. (2010). Bayes optimal multilabel classification via probabilistic classifier chains. In Proceedings of the 27th international conference on machine learning (ICML-10) (pp. 279-286).
