# NoChangeClassifier

Dummy classifier which returns the last class seen.

The predict\_one method will output the last class seen whilst predict\_proba\_one will return 1 for the last class seen and 0 for the others.

## Attributes

- **last\_class**

  The last class seen.
- **classes**

  The set of classes seen.

## Examples

Taken from example 2.1 from [this page](<https://www.cms.waikato.ac.nz/~abifet/book/chapter_2.html>).

```python
import pprint
from river import dummy

sentences = [
    ('glad happy glad', '+'),
    ('glad glad joyful', '+'),
    ('glad pleasant', '+'),
    ('miserable sad glad', '−')
]

model = dummy.NoChangeClassifier()

for sentence, label in sentences:
    model.learn_one(sentence, label)

new_sentence = 'glad sad miserable pleasant glad'
model.predict_one(new_sentence)
```

```text
'−'
```

```python
pprint.pprint(model.predict_proba_one(new_sentence))
```

```text
{'+': 0.0, '−': 1}
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
> - **x** — `object`
> - **kwargs** — `object`
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
