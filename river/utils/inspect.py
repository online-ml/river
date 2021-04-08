"""Utilities for inspecting a model's type.

Sometimes we need to check if a model can perform regression, classification, etc. However, for
some models the model's type is only known at runtime. For instance, we can't do
`isinstance(pipeline, base.Regressor)` or `isinstance(wrapper, base.Regressor)`. This submodule
thus provides utilities for determining an arbitrary model's type.

"""
from river import base, compose

# TODO: maybe all of this could be done by monkeypatching isintance for pipelines?


__all__ = [
    "extract_relevant",
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
        return extract_relevant(list(model.steps.values())[-1])  # look at last step
    return model


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
