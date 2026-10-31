# learn\_during\_predict

A context manager for fitting unsupervised steps during prediction.

Usually, unsupervised parts of a pipeline are updated during `learn_one`. However, in the case of online learning, it is possible to update them before, during the prediction step. This context manager allows you to do so.

This usually brings a slight performance improvement. But it is not done by default because it is not intuitive and is more difficult to test. It also means that you have to call `predict_one` before `learn_one` in order for the whole pipeline to be updated.

## Examples

Let's first see what methods are called if we just call `predict_one`.

```python
import io
import logging
import pandas as pd
from river import compose
from river import datasets
from river import linear_model
from river import preprocessing
from river import utils

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LinearRegression()
)

class_condition = lambda x: x.__class__.__name__ in ('StandardScaler', 'LinearRegression')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logs = io.StringIO()
sh = logging.StreamHandler(logs)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

with utils.log_method_calls(class_condition):
    for x, y in datasets.TrumpApproval().take(1):
        _ = model.predict_one(x)

print(logs.getvalue())
```

```text
StandardScaler.transform_one
LinearRegression.predict_one
```

Now let's use the context manager and see what methods get called.

```python
logs = io.StringIO()
sh = logging.StreamHandler(logs)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

with utils.log_method_calls(class_condition), compose.learn_during_predict():
    for x, y in datasets.TrumpApproval().take(1):
        _ = model.predict_one(x)

print(logs.getvalue())
```

```text
StandardScaler.learn_one
StandardScaler.transform_one
LinearRegression.predict_one
```

We can see that the scaler did not get updated before transforming the data.

This also works when working with mini-batches.

```python
logs = io.StringIO()
sh = logging.StreamHandler(logs)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

with utils.log_method_calls(class_condition):
    for x, y in datasets.TrumpApproval().take(1):
        _ = model.predict_many(pd.DataFrame([x]))
print(logs.getvalue())
```

```text
StandardScaler.transform_many
LinearRegression.predict_many
```

```python
logs = io.StringIO()
sh = logging.StreamHandler(logs)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

with utils.log_method_calls(class_condition), compose.learn_during_predict():
    for x, y in datasets.TrumpApproval().take(1):
        _ = model.predict_many(pd.DataFrame([x]))
print(logs.getvalue())
```

```text
StandardScaler.learn_many
StandardScaler.transform_many
LinearRegression.predict_many
```
