from sklearn import utils
import itertools


__all__ = ['iter_numpy', 'iter_pandas', 'iter_sklearn_dataset']


def iter_numpy(X, y=None, feature_names=None, shuffle=False, random_state=None):
    feature_names = list(range(X.shape[1])) if feature_names is None else feature_names
    rng = utils.check_random_state(random_state)

    if shuffle:
        order = rng.permutation(len(X))
        X, y = X[order], y if y is None else y[order]

    for x, yi in itertools.zip_longest(X, [] if y is None else y):
        yield {i: xi for i, xi in zip(feature_names, x)}, yi


def iter_sklearn_dataset(load_dataset, **kwargs):
    dataset = load_dataset()
    kwargs['X'] = dataset.data
    kwargs['y'] = dataset.target
    kwargs['feature_names'] = dataset.feature_names

    for x, yi in iter_numpy(**kwargs):
        yield x, yi


def iter_pandas(X, y=None, **kwargs):
    kwargs['feature_names'] = X.columns
    for x, yi in iter_numpy(X, y, **kwargs):
        yield x, yi
