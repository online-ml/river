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
        "abrupt_balanced": (52848, 16419025),
        "abrupt_imbalanced": (355275, 110043637),
        "gradual_balanced": (24150, 7503750),
        "gradual_imbalanced": (143323, 44371501),
        "incremental-abrupt_balanced": (79986, 24849436),
        "incremental-abrupt_imbalanced": (452044, 140004225),
        "incremental-reoccurring_balanced": (79986, 24849092),
        "incremental-reoccurring_imbalanced": (452044, 140004230),
        "incremental_balanced": (57018, 17713574),
        "incremental_imbalanced": (452044, 140004218),
        "out-of-control": (905145, 277777854),
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
            url=f"https://sites.labic.icmc.usp.br/vsouza/repository/river/INSECTS-{variant}_norm.arff",
            size=size,
            unpack=False,
        )
        self.variant = variant

    def _iter(self):
        return stream.iter_arff(self.path, target="class")

    @property
    def _repr_content(self):
        return {**super()._repr_content, "Variant": self.variant}
