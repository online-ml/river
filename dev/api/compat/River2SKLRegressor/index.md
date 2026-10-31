# River2SKLRegressor

Compatibility layer from River to scikit-learn for regression.

## Parameters

- **river\_estimator**

  *Type* → `base.Regressor`

## Methods

> [!NOTE]
>
> **fit**
>
> Fits to an entire dataset contained in memory.
>
> **Parameters**
>
> - **X**
> - **y**
>
> **Returns**
>
> self

> [!NOTE]
>
> **get\_metadata\_routing**
>
> Get metadata routing of this object.
>
> Please check :ref:`User Guide <metadata_routing>` on how the routing mechanism works.
>
> **Returns**
>
> MetadataRequest

> [!NOTE]
>
> **get\_params**
>
> Get parameters for this estimator.
>
> **Parameters**
>
> - **deep** — defaults to `True`
>
> **Returns**
>
> dict

> [!NOTE]
>
> **partial\_fit**
>
> Fits incrementally on a portion of a dataset.
>
> **Parameters**
>
> - **X**
> - **y**
>
> **Returns**
>
> self

> [!NOTE]
>
> **predict**
>
> Predicts the target of an entire dataset contained in memory.
>
> **Parameters**
>
> - **X**
>
> **Returns**
>
> `np.ndarray`: Predicted target values for each row of `X`.

> [!NOTE]
>
> **score**
>
> Return :ref:`coefficient of determination <r2_score>` on test data.
>
> The coefficient of determination, :math:`R^2`, is defined as :math:`(1 - \frac{u}{v})`, where :math:`u` is the residual sum of squares `((y_true - y_pred)** 2).sum()` and :math:`v` is the total sum of squares `((y_true - y_true.mean()) ** 2).sum()`. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of `y`, disregarding the input features, would get a :math:`R^2` score of 0.0.
>
> **Parameters**
>
> - **X**
> - **y**
> - **sample\_weight** — defaults to `None`
>
> **Returns**
>
> float

> [!NOTE]
>
> **set\_params**
>
> Set the parameters of this estimator.
>
> The method works on simple estimators as well as on nested objects (such as :class:`~sklearn.pipeline.Pipeline`). The latter have parameters of the form `<component>__<parameter>` so that it's possible to update each component of a nested object.
>
> **Parameters**
>
> - **params**
>
> **Returns**
>
> estimator instance

> [!NOTE]
>
> **set\_score\_request**
>
> Configure whether metadata should be requested to be passed to the `score` method.
>
> Note that this method is only relevant when this estimator is used as a sub-estimator within a :term:`meta-estimator` and metadata routing is enabled with `enable_metadata_routing=True` (see :func:`sklearn.set_config`). Please check the :ref:`User Guide <metadata_routing>` on how the routing mechanism works. The options for each parameter are: - `True`: metadata is requested, and passed to `score` if provided. The request is ignored if metadata is not provided. - `False`: metadata is not requested and the meta-estimator will not pass it to `score`. - `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it. - `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name. The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the existing request. This allows you to change the request for some parameters and not others. .. versionadded:: 1.3
>
> **Parameters**
>
> - **sample\_weight** — `Union[bool, NoneType, str]` — defaults to `$UNCHANGED$`
>
> **Returns**
>
> `River2SKLRegressor`: object
