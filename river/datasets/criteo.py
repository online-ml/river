from __future__ import annotations

from river import stream

from . import base


class CriteoAds(base.RemoteDataset):
    """Criteo Display Advertising Challenge.

    This is a 100,000-row sample of the Criteo Display Advertising Challenge
    dataset: real ad-impression logs where the goal is to predict whether an ad
    was clicked. Each row has a binary `click` target, 13 integer features (`I1`
    to `I13`, mostly anonymised counts, frequently missing) and 26 categorical
    features (`C1` to `C26`, hashed to opaque strings). The categorical fields are
    extremely high-cardinality, which makes this a natural fit for one-hot models
    such as `linear_model.AdPredictor`.

    Integer features are returned as `int` (or `None` when missing) and
    categorical features as `str` (or `None` when missing).

    References
    ----------
    [^1]: [Display Advertising Challenge - Criteo AI Lab](https://ailab.criteo.com/display-advertising-challenge-criteo/)
    [^2]: [Kaggle competition page](https://www.kaggle.com/c/criteo-display-ad-challenge)

    """

    def __init__(self):
        super().__init__(
            n_samples=100_000,
            n_features=39,
            task=base.BINARY_CLF,
            url="https://maxhalford.github.io/files/datasets/criteo_sample.csv.gz",
            size=8_749_440,
            unpack=False,
            filename="criteo_sample.csv.gz",
        )

    def _iter(self):
        int_cols = [f"I{i}" for i in range(1, 14)]
        cat_cols = [f"C{i}" for i in range(1, 27)]
        converters: dict = {"click": lambda x: x == "1"}
        converters.update({c: lambda x: int(x) if x else None for c in int_cols})
        converters.update({c: lambda x: x if x else None for c in cat_cols})

        return stream.iter_csv(self.path, target="click", converters=converters)
