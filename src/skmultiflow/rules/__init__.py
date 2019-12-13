"""
The :mod:`skmultiflow.rules` module includes rule-based learning methods.
"""

from .very_fast_decision_rules import VeryFastDecisionRulesClassifier
from .very_fast_decision_rules import VFDR   # remove in v0.7.0

__all__ = ["VeryFastDecisionRulesClassifier", "VFDR"]
