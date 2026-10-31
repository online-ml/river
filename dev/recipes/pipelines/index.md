# Pipelines

Pipelines are an integral part of River. A pipeline chains estimators in sequence: the first `n - 1` steps must be transformers, and the last step can be any estimator (regressor, classifier, clusterer, transformer, etc.).

For a hands-on walkthrough that builds a pipeline from scratch on a real dataset, see [The art of using pipelines](<https://riverml.xyz/dev/examples/the-art-of-using-pipelines/index.md>).

## Building a pipeline

There are three equivalent ways to create a pipeline:

```python
from river import compose, feature_extraction, linear_model, preprocessing

# Using the constructor
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    feature_extraction.PolynomialExtender(),
    linear_model.LinearRegression(),
)

# Using the | operator
model = (
    preprocessing.StandardScaler()
    | feature_extraction.PolynomialExtender()
    | linear_model.LinearRegression()
)

# Using |= incrementally
model = preprocessing.StandardScaler()
model |= feature_extraction.PolynomialExtender()
model |= linear_model.LinearRegression()
```

## How pipelines work

- **`learn_one(x, y)`** calls `learn_one` on each step sequentially, passing transformed data forward.
- **`predict_one(x)`** calls `transform_one` on the first `n - 1` steps, then `predict_one` on the last step.
- **`transform_one(x)`** calls `transform_one` on each transformer and returns the output of the last one.

You can inspect individual steps using bracket notation:

```python
from river import datasets

model = preprocessing.StandardScaler() | linear_model.LinearRegression()

for x, y in datasets.TrumpApproval().take(5):
    model.learn_one(x, y)

# Inspect the scaler's learned means
model["StandardScaler"].means
```

## TransformerUnion

Sometimes you want to apply multiple transformers in parallel and merge their outputs. Use `+` to create a [`compose.TransformerUnion`](<https://riverml.xyz/dev/api/compose/TransformerUnion>):

```python
model = (
    preprocessing.StandardScaler()
    | (feature_extraction.PolynomialExtender() + feature_extraction.RBFSampler())
    | linear_model.LinearRegression()
)

model
```

The `+` operator is shorthand for [`compose.TransformerUnion`](<https://riverml.xyz/dev/api/compose/TransformerUnion>)(...)\`. Both transformers receive the same input, and their outputs are merged into a single dict.

## Custom functions in pipelines

Use [`compose.FuncTransformer`](<https://riverml.xyz/dev/api/compose/FuncTransformer>) to include arbitrary functions as pipeline steps. Functions used with the `|` operator are automatically wrapped:

```python
def add_interaction(x):
    x["interaction"] = x["gallup"] * x["ipsos"]
    return x

model = (
    add_interaction
    | preprocessing.StandardScaler()
    | linear_model.LinearRegression()
)

model
```

## Learning during predict

In online learning, unsupervised steps (like `StandardScaler`) don't need the ground truth label to update. By default, they only update during `learn_one`. But you can make them update during `predict_one` too, using the [`compose.learn_during_predict`](<https://riverml.xyz/dev/api/compose/learn-during-predict>) context manager:

```python
model = preprocessing.StandardScaler() | linear_model.LinearRegression()

x, y = next(iter(datasets.TrumpApproval()))

# Without learn_during_predict: scaler doesn't update
model.predict_one(x)
print(f"After predict_one: means are all zero = {all(v == 0 for v in model['StandardScaler'].means.values())}")

# With learn_during_predict: scaler updates on predict
with compose.learn_during_predict():
    model.predict_one(x)
print(f"After learn_during_predict: means are all zero = {all(v == 0 for v in model['StandardScaler'].means.values())}")
```

Only unsupervised components update during predict — the supervised parts (like `LinearRegression`) still wait for `learn_one`.

This is especially useful when labeled data is scarce or delayed: the scaler keeps adapting to the data distribution even when ground truth hasn't arrived yet.

### Performance impact

The benefit grows as the fraction of labeled samples decreases:

```python
from contextlib import nullcontext
from river import metrics

def score(learn_during_predict, pct_labeled):
    dataset = datasets.TrumpApproval()
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    metric = metrics.MAE()
    n_learn = int(dataset.n_samples * pct_labeled)

    ctx = compose.learn_during_predict if learn_during_predict else nullcontext
    with ctx():
        for i, (x, y) in enumerate(dataset):
            metric.update(y, model.predict_one(x))
            if i < n_learn:
                model.learn_one(x, y)
    return metric.get()

for pct in [1.0, 0.5, 0.1]:
    off = score(False, pct)
    on = score(True, pct)
    print(f"{pct:>5.0%} labeled — MAE without: {off:8.2f}, with: {on:5.2f}")
```

When all samples are labeled, the effect is negligible. But with only 10% labeled data, learning during predict reduces the MAE by an order of magnitude.

## Key takeaways

- Build pipelines with `|` (sequential) and `+` (parallel union).
- `learn_one` updates all steps; `predict_one` transforms then predicts.
- Use [`compose.FuncTransformer`](<https://riverml.xyz/dev/api/compose/FuncTransformer>) or pass functions directly to `|` for custom logic.
- [`compose.learn_during_predict`](<https://riverml.xyz/dev/api/compose/learn-during-predict>)()\` lets unsupervised steps update at prediction time — powerful when labels are scarce.
