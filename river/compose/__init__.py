"""Model composition.

This module contains utilities for merging multiple modeling steps into a single pipeline. Although
pipelines are not the only way to process a stream of data, we highly encourage you to use them.

"""
from .func import FuncTransformer
from .grouper import Grouper
from .pipeline import Pipeline
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
    "Renamer",
    "Select",
    "SelectType",
    "Suffixer",
    "TargetTransformRegressor",
    "TransformerProduct",
    "TransformerUnion",
]
