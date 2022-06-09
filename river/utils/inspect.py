"""Utilities for inspecting a model's type.

Sometimes we need to check if a model can perform regression, classification, etc. However, for
some models the model's type is only known at runtime. For instance, we can't do
`isinstance(pipeline, base.Regressor)` or `isinstance(wrapper, base.Regressor)`. This submodule
thus provides utilities for determining an arbitrary model's type.

"""
import inspect

from river import base, compose

# TODO: maybe all of this could be done by monkeypatching isintance for pipelines?


__all__ = [
    "extract_relevant",
    "isanomalydetector",
    "isclassifier",
    "isregressor",
    "ismoclassifier",
    "ismoregressor",
    "isdriftdetector",
]


def extract_relevant(model: base.Estimator):
    """Extracts the relevant part of a model.

    Parameters
    ----------
    model

    """

    if isinstance(model, compose.Pipeline):
        return extract_relevant(model._last_step)
    return model


def isanomalydetector(model):
    """Checks weather or not the given model inherits from AnomalyDetector.

    Parameters
    ----------
    model

    Returns
    -------

    True if the input model object has inherited from AnomalyDetector else False.

    Examples
    --------

    >>> from river import anomaly
    >>> from river import utils

    >>> utils.inspect.isanomalydetector(model=anomaly.HalfSpaceTrees())
    True

    >>> utils.inspect.isanomalydetector(model=anomaly.OneClassSVM())
    True

    >>> utils.inspect.isanomalydetector(model=anomaly.GaussianScorer())
    False
    """
    model = extract_relevant(model)
    parent_classes = inspect.getmro(model.__class__)
    return any(cls.__name__ == "AnomalyDetector" for cls in parent_classes)


def isclassifier(model):
    return isinstance(extract_relevant(model), base.Classifier)


def isclusterer(model):
    return isinstance(extract_relevant(model), base.Clusterer)


def ismoclassifier(model):
    return isclassifier(model) and isinstance(
        extract_relevant(model), base.MultiOutputMixin
    )


def isregressor(model):
    return isinstance(extract_relevant(model), base.Regressor)


def istransformer(model):
    return isinstance(extract_relevant(model), base.Transformer)


def ismoregressor(model):
    return isregressor(model) and isinstance(
        extract_relevant(model), base.MultiOutputMixin
    )


def isdriftdetector(model):
    return isinstance(extract_relevant(model), base.DriftDetector)
