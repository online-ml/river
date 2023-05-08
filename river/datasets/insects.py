from __future__ import annotations

from river import stream

from . import base


class Insects(base.RemoteDataset):
    """Insects dataset.

    This dataset has different variants, which are:

    - abrupt_balanced
    - abrupt_imbalanced
    - gradual_balanced
    - gradual_imbalanced
    - incremental-abrupt_balanced
    - incremental-abrupt_imbalanced
    - incremental-reoccurring_balanced
    - incremental-reoccurring_imbalanced
    - incremental_balanced
    - incremental_imbalanced
    - out-of-control

    The number of samples and the difficulty change from one variant to another. The number of
    classes is always the same (6), except for the last variant (24).

    Parameters
    ----------
    variant
        Indicates which variant of the dataset to load.

    References
    ----------
    [^1]: [USP DS repository](https://sites.google.com/view/uspdsrepository)
    [^2]: [Souza, V., Reis, D.M.D., Maletzke, A.G. and Batista, G.E., 2020. Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. arXiv preprint arXiv:2005.00113.](https://arxiv.org/abs/2005.00113)

    """

    variant_sizes = {
        "abrupt_balanced": (52_848, 16_419_025),
        "abrupt_imbalanced": (355_275, 110_043_637),
        "gradual_balanced": (24_150, 7_503_750),
        "gradual_imbalanced": (143_323, 44_371_501),
        "incremental-abrupt_balanced": (79_986, 24_849_436),
        "incremental-abrupt_imbalanced": (452_044, 140_004_225),
        "incremental-reoccurring_balanced": (79_986, 24_849_092),
        "incremental-reoccurring_imbalanced": (452_044, 140_004_230),
        "incremental_balanced": (57_018, 17_713_574),
        "incremental_imbalanced": (452_044, 140_004_218),
        "out-of-control": (905_145, 277_777_854),
    }

    variants = list(variant_sizes.keys())

    def __init__(self, variant="abrupt_balanced"):
        try:
            n_samples, size = self.variant_sizes[variant]
        except KeyError:
            variants = "\n".join(f"- {v}" for v in self.variant_sizes)
            raise ValueError(f"Unknown variant, possible choices are:\n{variants}")
        n_classes = 24 if variant == "out-of-control" else 6

        super().__init__(
            n_classes=n_classes,
            n_samples=n_samples,
            n_features=33,
            task=base.MULTI_CLF,
            url=f"http://sites.labic.icmc.usp.br/vsouza/repository/creme/INSECTS-{variant}_norm.arff",
            size=size,
            unpack=False,
        )
        self.variant = variant

    def _iter(self):
        return stream.iter_arff(self.path, target="class")

    @property
    def _repr_content(self):
        return {**super()._repr_content, "Variant": self.variant}
