# SelectType

Selects features based on their type.

This is practical when you want to apply different preprocessing steps to different kinds of features. For instance, a common usecase is to apply a [`preprocessing.StandardScaler`](<https://riverml.xyz/dev/api/preprocessing/StandardScaler>) to numeric features and a [`preprocessing.OneHotEncoder`](<https://riverml.xyz/dev/api/preprocessing/OneHotEncoder>) to categorical features.

## Examples

```python
import numbers
from river import compose
from river import linear_model
from river import preprocessing

num = compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
cat = compose.SelectType(str) | preprocessing.OneHotEncoder()
model = (num + cat) | linear_model.LogisticRegression()
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update with a set of features `x`.
>
> A lot of transformers don't actually have to do anything during the `learn_one` step because they are stateless. For this reason the default behavior of this function is to do nothing. Transformers that however do something during the `learn_one` can override this method.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

> [!NOTE]
>
> **transform\_one**
>
> Transform a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, Any&#93;`: The transformed values.
