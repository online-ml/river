# StatisticRegressor

Dummy regressor that uses a univariate statistic to make predictions.

## Parameters

- **statistic**

  *Type* → `stats.base.Univariate`

## Examples

```python
from pprint import pprint
from river import dummy
from river import stats

sentences = [
    ('glad happy glad', 3),
    ('glad glad joyful', 3),
    ('glad pleasant', 2),
    ('miserable sad glad', -3)
]

model = dummy.StatisticRegressor(stats.Mean())

for sentence, score in sentences:
    model.learn_one(sentence, score)

new_sentence = 'glad sad miserable pleasant glad'
model.predict_one(new_sentence)
```

```text
1.25
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
>
> **Parameters**
>
> - **x** — `object`
> - **y** — `base.typing.RegTarget`

> [!NOTE]
>
> **predict\_one**
>
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x** — `object`
>
> **Returns**
>
> `base.typing.RegTarget | None`: The prediction.
