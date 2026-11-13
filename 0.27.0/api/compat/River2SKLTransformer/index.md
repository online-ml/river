# River2SKLTransformer

Compatibility layer from River to scikit-learn for transformation.

## Parameters

- **river\_estimator**

  *Type* → `base.Transformer`

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
> - **y** — defaults to `None`
>
> **Returns**
>
> self

> [!NOTE]
>
> **fit\_transform**
>
> Fit to data, then transform it.
>
> Fits transformer to `X` and `y` with optional parameters `fit_params` and returns a transformed version of `X`.
>
> **Parameters**
>
> - **X**
> - **y** — defaults to `None`
> - **fit\_params**
>
> **Returns**
>
> ndarray array of shape (n\_samples, n\_features\_new)

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
> - **y** — defaults to `None`
>
> **Returns**
>
> self

> [!NOTE]
>
> **set\_output**
>
> Set output container.
>
> See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py` for an example on how to use the API.
>
> **Parameters**
>
> - **transform** — defaults to `None`
>
> **Returns**
>
> estimator instance

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
> **transform**
>
> Predicts the target of an entire dataset contained in memory.
>
> **Parameters**
>
> - **X**
>
> **Returns**
>
> Transformed output.
