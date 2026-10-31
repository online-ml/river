# ComplementNB

Naive Bayes classifier for multinomial models.

Complement Naive Bayes model learns from occurrences between features such as word counts and discrete classes. ComplementNB is suitable for imbalance dataset. The input vector must contain positive values, such as counts or TF-IDF values.

## Parameters

- **alpha**

  *Default* → `1.0`

  Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

## Attributes

- **class\_counts** (`collections.Counter`)

  Number of times each class has been seen.
- **feature\_counts** (`collections.defaultdict`)

  Total frequencies per feature and class.
- **class\_totals** (`collections.Counter`)

  Total frequencies per class.

## Examples

```python
import pandas as pd
from river import compose
from river import feature_extraction
from river import naive_bayes

docs = [
    ("Chinese Beijing Chinese", "yes"),
    ("Chinese Chinese Shanghai", "yes"),
    ("Chinese Macao", "maybe"),
    ("Tokyo Japan Chinese", "no")
]

model = compose.Pipeline(
    ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
    ("nb", naive_bayes.ComplementNB(alpha=1))
)

for sentence, label in docs:
    model.learn_one(sentence, label)

model["nb"].p_class("yes")
```

```text
0.5
```

```python
model["nb"].p_class("no")
```

```text
0.25
```

```python
model["nb"].p_class("maybe")
```

```text
0.25
```

```python
model.predict_proba_one("test")
```

```text
{'yes': 0.275, 'maybe': 0.375, 'no': 0.35}
```

```python
model.predict_one("test")
```

```text
'maybe'
```

You can train the model and make predictions in mini-batch mode using the class methods `learn_many` and `predict_many`.

```python
df_docs = pd.DataFrame(docs, columns = ["docs", "y"])

X = pd.Series([
   "Chinese Beijing Chinese",
   "Chinese Chinese Shanghai",
   "Chinese Macao",
   "Tokyo Japan Chinese"
])

y = pd.Series(["yes", "yes", "maybe", "no"])

model = compose.Pipeline(
    ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
    ("nb", naive_bayes.ComplementNB(alpha=1))
)

model.learn_many(X, y)

unseen = pd.Series(["Taiwanese Taipei", "Chinese Shanghai"])

model.predict_proba_many(unseen)
```

```text
      maybe        no       yes
0  0.415129  0.361624  0.223247
1  0.248619  0.216575  0.534807
```

```python
model.predict_many(unseen)
```

```text
0    maybe
1      yes
dtype: object
```

## Methods

> [!NOTE]
>
> **joint\_log\_likelihood**
>
> Computes the joint log likelihood of input features.
>
> **Parameters**
>
> - **x** — `dict`
>
> **Returns**
>
> `float`: Mapping between classes and joint log likelihood.

> [!NOTE]
>
> **joint\_log\_likelihood\_many**
>
> Computes the joint log likelihood of input features.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`
>
> **Returns**
>
> `pd.DataFrame`: Input samples joint log likelihood.

> [!NOTE]
>
> **learn\_many**
>
> Learn from a batch of count vectors.
>
> **Parameters**
>
> - **X** — `pd.DataFrame`
> - **y** — `pd.Series`

> [!NOTE]
>
> **learn\_one**
>
> Updates the model with a single observation.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.ClfTarget`

> [!NOTE]
>
> **p\_class**

> [!NOTE]
>
> **p\_class\_many**

> [!NOTE]
>
> **predict\_many**
>
> Predict the outcome for each given sample.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `pd.Series`: The predicted labels.

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `base.typing.ClfTarget | None`: The predicted label.

> [!NOTE]
>
> **predict\_proba\_many**
>
> Return probabilities using the log-likelihoods in mini-batchs setting.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

> [!NOTE]
>
> **predict\_proba\_one**
>
> Return probabilities using the log-likelihoods.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

## References

- [Rennie, J.D., Shih, L., Teevan, J. and Karger, D.R., 2003. Tackling the poor assumptions of naive bayes text classifiers. In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 616-623)](<https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf>)
- [StackExchange discussion](<https://stats.stackexchange.com/questions/126009/complement-naive-bayes>)
