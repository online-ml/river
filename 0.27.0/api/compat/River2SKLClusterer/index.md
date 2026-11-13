# River2SKLClusterer

Compatibility layer from River to scikit-learn for clustering.

## Parameters

- **river\_estimator**

  *Type* → `base.Clusterer`

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
> **fit\_predict**
>
> Perform clustering on `X` and returns cluster labels.
>
> **Parameters**
>
> - **X**
> - **y** — defaults to `None`
> - **kwargs**
>
> **Returns**
>
> ndarray of shape (n\_samples,), dtype=np.int64

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
> Transformed output.

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
