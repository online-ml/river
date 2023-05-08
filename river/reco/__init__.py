"""Recommender systems module.

Recommender systems (recsys for short) is a large topic. This module is far from comprehensive. It
simply provides models which can contribute towards building a recommender system.

A typical recommender system is made up of a retrieval phase, followed by a ranking phase. The
output of the retrieval phase is a shortlist of the catalogue of items. The items in the shortlist
are then usually ranked according to the expected preference the user will have for each item. This
module focuses on the ranking phase.

Models which inherit from the `Ranker` class have a `rank` method. This allows sorting a set of
items for a given user. Each model also has a `learn_one(user, item, y, context)` which allows
learning user preferences. The `y` parameter is a reward value, the nature of which depends is
specific to each and every recommendation task. Typically the reward is a number or a boolean
value. It is up to the user to determine how to translate a user session into training data.

"""
from __future__ import annotations

from . import base
from .baseline import Baseline
from .biased_mf import BiasedMF
from .funk_mf import FunkMF
from .normal import RandomNormal

__all__ = ["base", "Baseline", "BiasedMF", "FunkMF", "RandomNormal"]
