def iter_vaex(X, y=None, features=None, **kwargs):
    """Yields rows from a ``vaex.DataFrame``.

    Parameters:
        X (vaex.DataFrame): A vaex DataFrame housing the training featuers.
        y (string or vaex.Expression): The column or expression containing the target variable.
        features (list of strings or vaex.Expressions): A list of features used for training.
        If None, all columns in ``X`` will be used. Features specifying in ``y`` are ignored.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    """

    from vaex.utils import _ensure_strings_from_expressions, _ensure_list

    features = _ensure_strings_from_expressions(features)
    feature_names = features or X.get_column_names()

    if y:
        y = _ensure_strings_from_expressions(y)
        y = _ensure_list(y)
        feature_names = [feat for feat in feature_names if feat not in y]

    multioutput = len(y) > 1

    if multioutput:
        for i in range(len(X)):
            yield (
                {key: X.evaluate(key, i, i + 1)[0] for key in feature_names},
                {key: X.evaluate(key, i, i + 1)[0] for key in y}
            )

    else:

        for i in range(len(X)):
            yield (
                {key: X.evaluate(key, i, i + 1)[0] for key in feature_names},
                X.evaluate(y[0], i, i + 1)[0]
            )
