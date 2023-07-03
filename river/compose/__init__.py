"""Model composition.

This module contains utilities for merging multiple modeling steps into a single pipeline. Although
pipelines are not the only way to process a stream of data, we highly encourage you to use them.

"""
from __future__ import annotations

from .func import FuncTransformer
from .grouper import Grouper
from .pipeline import Pipeline, learn_during_predict
from .product import TransformerProduct
from .renamer import Prefixer, Renamer, Suffixer
from .select import Discard, Select, SelectType
from .target_transform import TargetTransformRegressor
from .union import TransformerUnion

__all__ = [
    "Discard",
    "FuncTransformer",
    "Grouper",
    "Pipeline",
    "Prefixer",
    "pure_inference_mode",
    "Renamer",
    "Select",
    "SelectType",
    "Suffixer",
    "TargetTransformRegressor",
    "TransformerProduct",
    "TransformerUnion",
    "learn_during_predict",
]
