from river import base


def clone(estimator: base.Estimator):
    """Return a fresh estimator with the same parameters.

    Essentially, this acts as if you reinitialized the estimator with the same initial set of
    parameters.

    Parameters:
        estimator: The estimator to clone.

    """
    return estimator._set_params()
