# PriorClassifier

Dummy classifier which uses the prior distribution.

The `predict_one` method will output the most common class whilst `predict_proba_one` will return the normalized class counts.

## Attributes

- **counts** (`collections.Counter`)

  Class counts.
- **n** (`int`)

  Total number of seen instances.

## Examples

Taken from example 2.1 from [this page](<https://www.cms.waikato.ac.nz/~abifet/book/chapter_2.html>)

```python
from river import dummy

sentences = [
    ('glad happy glad', '+'),
    ('glad glad joyful', '+'),
    ('glad pleasant', '+'),
    ('miserable sad glad', '−')
]

model = dummy.PriorClassifier()

for sentence, label in sentences:
    model.learn_one(sentence, label)

new_sentence = 'glad sad miserable pleasant glad'
model.predict_one(new_sentence)
```

```text
'+'
```

```python
model.predict_proba_one(new_sentence)
```

```text
{'+': 0.75, '−': 0.25}
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x** — `object`
> - **y** — `base.typing.ClfTarget`

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
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x** — `object`
> - **kwargs** — `object`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

## References

- [Krichevsky–Trofimov estimator](<https://en.wikipedia.org/wiki/Krichevsky–Trofimov_estimator>)
