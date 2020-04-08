
import sklearn.utils

from creme import base
from creme import stream


def iter_sklearn_dataset(dataset: 'sklearn.utils.Bunch', **kwargs) -> base.typing.Stream:
    """Yields rows from one of the datasets provided by scikit-learn.

    Parameters:
        dataset: A scikit-learn dataset.

    """
    kwargs['X'] = dataset.data
    kwargs['y'] = dataset.target
    try:
        kwargs['feature_names'] = dataset.feature_names
    except AttributeError:
        pass

    yield from stream.iter_array(**kwargs)
