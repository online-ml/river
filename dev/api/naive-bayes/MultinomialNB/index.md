# MultinomialNB

Naive Bayes classifier for multinomial models.

Multinomial Naive Bayes model learns from occurrences between features such as word counts and discrete classes. The input vector must contain positive values, such as counts or TF-IDF values.

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
    ("nb", naive_bayes.MultinomialNB(alpha=1))
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
{'yes': 0.413, 'maybe': 0.310, 'no': 0.275}
```

```python
model.predict_one("test")
```

```text
'yes'
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
    ("nb", naive_bayes.MultinomialNB(alpha=1))
)

model.learn_many(X, y)

unseen = pd.Series(["Taiwanese Taipei", "Chinese Shanghai"])

model.predict_proba_many(unseen)
```

```text
      maybe        no       yes
0  0.373272  0.294931  0.331797
1  0.160396  0.126733  0.712871
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
> - **X** — `IntoDataFrame`
>
> **Returns**
>
> `IntoDataFrame`: Input samples joint log likelihood.

> [!NOTE]
>
> **learn\_many**
>
> Learn from a batch of count vectors.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries`

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
> **p\_feature\_given\_class**

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

- [Naive Bayes text classification](<https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html>)
