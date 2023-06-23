"""Utilities for inspecting a model's type.

Sometimes we need to check if a model can perform regression, classification, etc. However, for
some models the model's type is only known at runtime. For instance, we can't do
`isinstance(pipeline, base.Regressor)` or `isinstance(wrapper, base.Regressor)`. This submodule
thus provides utilities for determining an arbitrary model's type.

"""
from __future__ import annotations

import inspect

from river import base

# TODO: maybe all of this could be done by monkeypatching isintance for pipelines?


__all__ = [
    "extract_relevant",
    "isactivelearner",
    "isanomalydetector",
    "isanomalyfilter",
    "isclassifier",
    "isclusterer",
    "isdriftdetector",
    "ismoclassifier",
    "ismoregressor",
    "isregressor",
    "istransformer",
]


def extract_relevant(model: base.Estimator):
    """Extracts the relevant part of a model.

    Parameters
    ----------
    model

    """

    if ischildobject(obj=model, class_name="Pipeline"):
        return extract_relevant(model._last_step)  # type: ignore[attr-defined]
    return model


def ischildobject(obj: object, class_name: str) -> bool:
    """Checks weather or not the given object inherits from a class with the given class name.

    Workaround isinstance function to not have to import modules defining classes and become
    dependent on them. class_name is case-sensitive.

    Examples
    --------

    >>> from river import anomaly
    >>> from river import utils

    >>> class_name = "AnomalyDetector"

    >>> utils.inspect.ischildobject(obj=anomaly.HalfSpaceTrees(), class_name=class_name)
    True

    >>> utils.inspect.ischildobject(obj=anomaly.OneClassSVM(), class_name=class_name)
    True

    >>> utils.inspect.ischildobject(obj=anomaly.GaussianScorer(), class_name=class_name)
    False

    """
    parent_classes = inspect.getmro(obj.__class__)
    return any(cls.__name__ == class_name for cls in parent_classes)


def isanomalydetector(model):
    model = extract_relevant(model)
    return ischildobject(obj=model, class_name="AnomalyDetector")


def isanomalyfilter(model):
    model = extract_relevant(model)
    return ischildobject(obj=model, class_name="AnomalyFilter")


def isclassifier(model):
    return isinstance(extract_relevant(model), base.Classifier)


def isclusterer(model):
    return isinstance(extract_relevant(model), base.Clusterer)


def ismoclassifier(model):
    return isinstance(extract_relevant(model), base.MultiLabelClassifier)


def isregressor(model):
    return isinstance(extract_relevant(model), base.Regressor)


def istransformer(model):
    return isinstance(extract_relevant(model), base.Transformer)


def ismoregressor(model):
    return isinstance(extract_relevant(model), base.MultiTargetRegressor)


def isdriftdetector(model):
    return isinstance(extract_relevant(model), base.DriftDetector)


def isactivelearner(model):
    from river import active

    return isinstance(extract_relevant(model), active.base.ActiveLearningClassifier)
