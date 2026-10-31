# expand\_param\_grid

Expands a grid of parameters.

This method can be used to generate a list of model parametrizations from a dictionary where each parameter is associated with a list of possible parameters. In other words, it expands a grid of parameters.

Typically, this method can be used to create copies of a given model with different parameter choices. The models can then be used as part of a model selection process, such as a `selection.SuccessiveHalvingClassifier` or a `selection.EWARegressor`.

The syntax for the parameter grid is quite flexible. It allows nesting parameters and can therefore be used to generate parameters for a pipeline.

## Parameters

- **model**

  *Type* → `T`
- **grid**

  *Type* → `Mapping[str, Params]`

  The grid of parameters to expand. The provided dictionary can be nested. The only requirement is that the values at the leaves need to be lists.

## Examples

As an initial example, we can expand a grid of parameters for a single model.

```python
from river import linear_model
from river import optim
from river import utils

model = linear_model.LinearRegression()

grid = {'optimizer': [optim.SGD(.1), optim.SGD(.01), optim.SGD(.001)]}
models = utils.expand_param_grid(model, grid)
len(models)
```

```text
3
```

```python
models[0]
```

```text
LinearRegression (
  optimizer=SGD (
    lr=Constant (
      learning_rate=0.1
    )
  )
  loss=Squared ()
  l2=0.
  l1=0.
  intercept_init=0.
  intercept_lr=Constant (
    learning_rate=0.01
  )
  clip_gradient=1e+12
  initializer=Zeros ()
)
```

You can expand parameters for multiple choices like so:

```python
grid = {
    'optimizer': [
        (optim.SGD, {'lr': [.1, .01, .001]}),
        (optim.Adam, {'lr': [.1, .01, .01]})
    ]
}
models = utils.expand_param_grid(model, grid)
len(models)
```

```text
6
```

You may specify a grid of parameters for a pipeline via nesting:

```python
from river import feature_extraction

model = (
    feature_extraction.BagOfWords() |
    linear_model.LinearRegression()
)

grid = {
    'BagOfWords': {
        'strip_accents': [False, True]
    },
    'LinearRegression': {
        'optimizer': [
            (optim.SGD, {'lr': [.1, .01]}),
            (optim.Adam, {'lr': [.1, .01]})
        ]
    }
}

models = utils.expand_param_grid(model, grid)
len(models)
```

```text
8
```
