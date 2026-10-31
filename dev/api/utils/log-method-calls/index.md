# log\_method\_calls

A context manager to log method calls.

All method calls will be logged by default. This behavior can be overridden by passing filtering functions.

## Parameters

- **class\_condition**

  *Type* → `typing.Callable[[typing.Any], bool] | None`

  *Default* → `None`

  A function which determines if a class should be logged or not.
- **method\_condition**

  *Type* → `typing.Callable[[typing.Any], bool] | None`

  *Default* → `None`

  A function which determines if a method should be logged or not.

## Examples

```python
import io
import logging
from river import anomaly
from river import compose
from river import datasets
from river import preprocessing
from river import utils

model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.HalfSpaceTrees(seed=42)
)

class_condition = lambda x: x.__class__.__name__ in ('MinMaxScaler', 'HalfSpaceTrees')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logs = io.StringIO()
sh = logging.StreamHandler(logs)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

with utils.log_method_calls(class_condition):
    for x, y in datasets.CreditCard().take(1):
        score = model.score_one(x)
        model.learn_one(x)

print(logs.getvalue())
```

```text
MinMaxScaler.transform_one
HalfSpaceTrees.score_one
MinMaxScaler.learn_one
MinMaxScaler.transform_one
HalfSpaceTrees.learn_one
```

```python
logs.close()
```
