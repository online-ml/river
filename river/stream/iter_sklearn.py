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

    >>> dataset = datasets.load_boston()

    >>> for xi, yi in stream.iter_sklearn_dataset(dataset):
    ...     pprint.pprint(xi)
    ...     print(yi)
    ...     break
    {'AGE': 65.2,
        'B': 396.9,
        'CHAS': 0.0,
        'CRIM': 0.00632,
        'DIS': 4.09,
        'INDUS': 2.31,
        'LSTAT': 4.98,
        'NOX': 0.538,
        'PTRATIO': 15.3,
        'RAD': 1.0,
        'RM': 6.575,
        'TAX': 296.0,
        'ZN': 18.0}
    24.0

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
