# iter\_progressive\_val\_score

Evaluates the performance of a model on a streaming dataset and yields results.

This does exactly the same as [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>). The only difference is that this function returns an iterator, yielding results at every step. This can be useful if you want to have control over what you do with the results. For instance, you might want to plot the results.

## Parameters

- **dataset**

  *Type* → `base.typing.Dataset`

  The stream of observations against which the model will be evaluated. Each element is an `(x, y)` pair or an `(x, y, kwargs)` triple where `kwargs` is a `dict` of extra parameters passed to `learn_one`. To supply per-sample weights, include a `"w"` key in `kwargs`, e.g. `(x, y, {"w": 2.0})`. The weight is forwarded to `learn_one` for models that accept a `w` parameter (e.g. `linear_model.LogisticRegression`).
- **model**

  The model to evaluate.
- **metric**

  *Type* → `metrics.base.Metric`

  The metric used to evaluate the model's predictions.
- **moment**

  *Type* → `str | typing.Callable | None`

  *Default* → `None`

  The attribute used for measuring time. If a callable is passed, then it is expected to take as input a `dict` of features. If `None`, then the observations are implicitly timestamped in the order in which they arrive.
- **delay**

  *Type* → `str | int | dt.timedelta | typing.Callable | None`

  *Default* → `None`

  The amount to wait before revealing the target associated with each observation to the model. This value is expected to be able to sum with the `moment` value. For instance, if `moment` is a `datetime.date`, then `delay` is expected to be a `datetime.timedelta`. If a callable is passed, then it is expected to take as input a `dict` of features and the target. If a `str` is passed, then it will be used to access the relevant field from the features. If `None` is passed, then no delay will be used, which leads to doing standard online validation, and `moment` is ignored.
- **step**

  *Default* → `1`

  Iteration number at which to yield results. This only takes into account the predictions, and not the training steps.
- **measure\_time**

  *Default* → `False`

  Whether or not to measure the elapsed time.
- **measure\_memory**

  *Default* → `False`

  Whether or not to measure the memory usage of the model.
- **yield\_predictions**

  *Default* → `False`

  Whether or not to include predictions. If step is 1, then this is equivalent to yielding the predictions at every iterations. Otherwise, not all predictions will be yielded.

## Examples

Take the following model:

```python
from river import linear_model
from river import preprocessing

model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression()
)
```

We can evaluate it on the `Phishing` dataset as so:

```python
from river import datasets
from river import evaluate
from river import metrics

steps = evaluate.iter_progressive_val_score(
    model=model,
    dataset=datasets.Phishing(),
    metric=metrics.ROCAUC(),
    step=200
)

for step in steps:
    print(step)
```

```text
{'ROCAUC': ROCAUC: 90.20%, 'Step': 200}
{'ROCAUC': ROCAUC: 92.25%, 'Step': 400}
{'ROCAUC': ROCAUC: 93.23%, 'Step': 600}
{'ROCAUC': ROCAUC: 94.05%, 'Step': 800}
{'ROCAUC': ROCAUC: 94.79%, 'Step': 1000}
{'ROCAUC': ROCAUC: 95.07%, 'Step': 1200}
{'ROCAUC': ROCAUC: 95.07%, 'Step': 1250}
```

The `yield_predictions` parameter can be used to include the predictions in the results:

```python
import itertools

steps = evaluate.iter_progressive_val_score(
    model=model,
    dataset=datasets.Phishing(),
    metric=metrics.ROCAUC(),
    step=1,
    yield_predictions=True
)

for step in itertools.islice(steps, 100, 105):
   print(step)
```

```text
{'ROCAUC': ROCAUC: 94.68%, 'Step': 101, 'Prediction': {False: 0.966..., True: 0.033...}}
{'ROCAUC': ROCAUC: 94.75%, 'Step': 102, 'Prediction': {False: 0.035..., True: 0.964...}}
{'ROCAUC': ROCAUC: 94.82%, 'Step': 103, 'Prediction': {False: 0.043..., True: 0.956...}}
{'ROCAUC': ROCAUC: 94.89%, 'Step': 104, 'Prediction': {False: 0.816..., True: 0.183...}}
{'ROCAUC': ROCAUC: 94.96%, 'Step': 105, 'Prediction': {False: 0.041..., True: 0.958...}}
```

## References

- [Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation](<http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf>)
- [Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30](<https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf>)
