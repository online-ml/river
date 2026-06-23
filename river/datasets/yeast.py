from __future__ import annotations

from river import stream

from . import base


class Yeast(base.FileDataset):
    """Yeast multi-label gene-function dataset.

    Micro-array expression and phylogenetic profiles for 2417 yeast genes, each
    annotated with a subset of 14 functional categories. This is the canonical
    multi-label version from the MULAN library; it matches OpenML dataset id
    40597 (`fetch_openml('yeast', version=4)`).

    References
    ----------
    [^1]: [Elisseeff, A. and Weston, J., 2002. A kernel method for multi-labelled classification. NeurIPS.](https://papers.nips.cc/paper/2001/hash/39dcaf7a053dc372fbc391d4e6b5d693-Abstract.html)
    [^2]: [Tsoumakas, G., Spyromitros-Xioufis, E., Vilcek, J. and Vlahavas, I., 2011. MULAN: A Java library for multi-label learning. JMLR.](https://jmlr.org/papers/v12/tsoumakas11a.html)

    """

    _FEATURES = tuple(f"Att{i}" for i in range(1, 104))
    _LABELS = tuple(f"Class{i}" for i in range(1, 15))

    def __init__(self):
        super().__init__(
            n_samples=2_417,
            n_features=len(self._FEATURES),
            n_outputs=len(self._LABELS),
            task=base.MO_BINARY_CLF,
            filename="yeast.csv.gz",
        )

    def __iter__(self):
        to_bool = lambda x: x == "1"  # noqa: E731
        converters: dict = {feat: float for feat in self._FEATURES}
        for label in self._LABELS:
            converters[label] = to_bool
        return stream.iter_csv(
            self.path,
            target=list(self._LABELS),
            converters=converters,
        )
