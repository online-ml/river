from __future__ import annotations

import numpy as np

from river import base
from river._river_rust.feature_hashing import feature_hash

__all__ = ["FeatureHasher"]


class FeatureHasher(base.Transformer):
    """Implements the hashing trick.

    Each pair of (name, value) features is hashed into a random integer in `[0, n_features)`,
    using the signed 32-bit MurmurHash3 of the feature's token. String values are hashed as
    `"name=value"` tokens and contribute `1`; numeric values are hashed under `"name"` and
    contribute the value itself.

    The hashing is performed in Rust, so the whole transform of an example happens in a single
    native call.

    Parameters
    ----------
    n_features
        The number by which each hash will be moduloed by.
    alternate_sign
        When `True` (the default), the sign bit of the hash is used to negate half of the
        contributions. This keeps the expected value of each bucket at zero, so hash collisions
        between unrelated features tend to cancel out rather than accumulate, which is especially
        helpful for small `n_features`. This matches scikit-learn's `FeatureHasher`.
    seed
        Set the seed to produce identical results. When `None`, a random seed is drawn, so two
        instances will hash features to different buckets.

    Examples
    --------

    >>> import river

    >>> hasher = river.preprocessing.FeatureHasher(n_features=10, seed=42)

    >>> X = [
    ...     {'dog': 1, 'cat': 2, 'elephant': 4},
    ...     {'dog': 2, 'run': 5}
    ... ]
    >>> for x in X:
    ...     print(hasher.transform_one(x))
    {5: -3, 7: 2}
    {5: 2, 9: -5}

    References
    ----------
    [^1]: [Wikipedia article on feature vectorization using the hashing trick](https://www.wikiwand.com/en/Feature_hashing#/Feature_vectorization_using_hashing_trick)
    [^2]: [Weinberger et al. (2009), Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206)

    """

    def __init__(self, n_features=1048576, seed: int | None = None, alternate_sign: bool = True):
        self.n_features = n_features
        self.seed = seed
        self.alternate_sign = alternate_sign
        # MurmurHash3 takes a 32-bit seed. Deriving it through NumPy preserves the previous
        # behaviour: a fixed `seed` is reproducible, while `seed=None` draws a fresh seed.
        self._seed = int(np.random.RandomState(seed).randint(0, 2**32, dtype=np.uint32))

    def transform_one(self, x):
        return feature_hash(x, self.n_features, self._seed, self.alternate_sign)
