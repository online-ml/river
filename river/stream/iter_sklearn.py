import pandas as pd
import sklearn.utils

from river import base, stream


def iter_sklearn_dataset(
    dataset: "sklearn.utils.Bunch", **kwargs
) -> base.typing.Stream:
    """Iterates rows from one of the datasets provided by scikit-learn.

    This allows you to use any dataset from [scikit-learn's `datasets` module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets). For instance, you can use the `fetch_openml` function to get access to all of the
    datasets from the OpenML website.

    Parameters
    ----------
    dataset
        A scikit-learn dataset.
    kwargs
        Extra keyword arguments are passed to the underlying call to `stream.iter_array`.

    Examples
    --------

    >>> import pprint
    >>> from sklearn import datasets
    >>> from river import stream

    >>> dataset = datasets.load_diabetes()

    >>> for xi, yi in stream.iter_sklearn_dataset(dataset):
    ...     pprint.pprint(xi)
    ...     print(yi)
    ...     break
    {'age': 0.0380759064334241,
     'bmi': 0.0616962065186885,
     'bp': 0.0218723549949558,
     's1': -0.0442234984244464,
     's2': -0.0348207628376986,
     's3': -0.0434008456520269,
     's4': -0.00259226199818282,
     's5': 0.0199084208763183,
     's6': -0.0176461251598052,
     'sex': 0.0506801187398187}
    151.0

    """
    kwargs["X"] = dataset.data
    kwargs["y"] = dataset.target
    try:
        kwargs["feature_names"] = dataset.feature_names
    except AttributeError:
        pass

    if isinstance(kwargs["X"], pd.DataFrame):
        yield from stream.iter_pandas(**kwargs)
    else:
        yield from stream.iter_array(**kwargs)
