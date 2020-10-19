"""Prototype-based methods.
These methods rely on typical representatives (prototypes) to extract
information from the data. Prototypes are identified from observed data. New
(unobserved) is then compared against the prototypes by means of a distance
metric.
"""

from .robust_soft_lvq import RSLVQClassifier

__all__ = ['RSLVQClassifier']
