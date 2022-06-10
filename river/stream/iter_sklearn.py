import pandas as pd
import sklearn.utils

from river import base, stream


def iter_sklearn_dataset(dataset: "sklearn.utils.Bunch", **kwargs) -> base.typing.Stream:
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
    {'age': 0.038075906433423026,
     'bmi': 0.061696206518683294,
     'bp': 0.0218723855140367,
     's1': -0.04422349842444599,
     's2': -0.03482076283769895,
     's3': -0.04340084565202491,
     's4': -0.002592261998183278,
     's5': 0.019907486170462722,
     's6': -0.01764612515980379,
     'sex': 0.05068011873981862}
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
