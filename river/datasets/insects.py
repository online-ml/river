from __future__ import annotations

from river import stream

from . import base


class Insects(base.RemoteDataset):
    """Insects dataset.

    This dataset has different variants for concept drift evaluation:

    - abrupt_balanced
    - abrupt_imbalanced
    - gradual_balanced
    - gradual_imbalanced
    - incremental_abrupt_balanced
    - incremental_reoccurring_balanced
    - incremental_balanced


    The number of samples and the difficulty change from one variant to another. The number of
    classes is always the same (6), except for the last variant (24).

    Parameters
    ----------
    variant
        Indicates which variant of the dataset to load. Defaults to "abrupt_balanced".

    References
    ----------
    [^1]: [USP DS repository](https://sites.google.com/view/uspdsrepository)
    [^2]: [Souza, V., Reis, D.M.D., Maletzke, A.G. and Batista, G.E., 2020. Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. arXiv preprint arXiv:2005.00113.](https://arxiv.org/abs/2005.00113)

    """

    variant_configs = {
        "abrupt_balanced": {
            "n_samples": 52_848,
            "size": 14_151_769,
            "url": "https://drive.google.com/uc?export=download&id=1WQoIuuVgiuXfzv4kvao6XuLQG37V923O&confirm=t",
            "filename": "abrupt_balanced.csv",
        },
        "abrupt_imbalanced": {
            "n_samples": 355_275,
            "size": 94_893_622,
            "url": "https://drive.google.com/uc?export=download&id=1Z9W2_mwawobXeTihZDTDnFOCtfdgmgDa&confirm=t",
            "filename": "abrupt_imbalanced.csv",
        },
        "gradual_balanced": {
            "n_samples": 24_150,
            "size": 6_474_831,
            "url": "https://drive.google.com/uc?export=download&id=1fepYkDxwMbuoRUaG_fsymSzkuapS4vJp&confirm=t",
            "filename": "gradual_balanced.csv",
        },
        "gradual_imbalanced": {
            "n_samples": 143_323,
            "size": 38_339_554,
            "url": "https://drive.google.com/uc?export=download&id=1_WJ7lxK2sx1i6kq9Nqv4KrjT0ZXB0QbC&confirm=t",
            "filename": "gradual_imbalanced.csv",
        },
        "incremental_abrupt_balanced": {
            "n_samples": 79_986,
            "size": 21_421_452,
            "url": "https://drive.google.com/uc?export=download&id=1-J5WIBN8_F_tomdcrOaiLCxk9nzxtFsf&confirm=t",
            "filename": "incremental_abrupt_balanced.csv",
        },
        # "incremental_abrupt_imbalanced": {
        #    "n_samples": 452_044,
        #    "size": 140_004_225,
        #    "url": "https://drive.google.com/uc?export=download&id=1M6QfsernUlM0qvqXdbo9bPYAsHCcfuhb&confirm=t",
        #    "filename": "incremental_abrupt_imbalanced.csv",
        # },
        "incremental_reoccurring_balanced": {
            "n_samples": 79_986,
            "size": 21_433_047,
            "url": "https://drive.google.com/uc?export=download&id=1mSKTSsxzYMjdV005AJqrcMGajuu7dUfW&confirm=t",
            "filename": "incremental_reoccurring_balanced.csv",
        },
        # "incremental_reoccurring_imbalanced": {
        #    "n_samples": 452_044,
        #    "size": 140_004_230,
        #    "url": "https://drive.google.com/uc?export=download&id=1aSqdxvZvug-SwQw5NLY_9nTPhhEjB2ig&confirm=t",
        #    "filename": "incremental_reoccurring_imbalanced.csv",
        # },
        "incremental_balanced": {
            "n_samples": 57_018,
            "size": 15_258_997,
            "url": "https://drive.google.com/uc?export=download&id=1tKQ2KL4m-ACHCVKUDLFPrM4cyhioiOpu&confirm=t",
            "filename": "incremental_balanced.csv",
        },
        # "incremental_imbalanced": {
        #    "n_samples": 452_044,
        #    "size": 140_004_218,
        #    "url": "https://drive.google.com/uc?export=download&id=1K3vp0EjA4FPDeSgffiBe4CMosbB4CKbL&confirm=t",
        #    "filename": "incremental_imbalanced.csv",
        # }
    }

    def __init__(self, variant="abrupt_balanced"):
        if variant not in self.variant_configs:
            variants = "\n".join(f"- {v}" for v in self.variant_configs)
            raise ValueError(f"Unknown variant, possible choices are:\n{variants}")

        config = self.variant_configs[variant]
        n_samples = config["n_samples"]
        size = config["size"]
        url = config["url"]
        filename = config["filename"]
        n_classes = 24 if variant == "out-of-control" else 6

        super().__init__(
            n_classes=n_classes,
            n_samples=n_samples,
            n_features=33,
            task=base.MULTI_CLF,
            url=url,
            size=size,
            unpack=False,
            filename=filename,
        )
        self.variant = variant

    def _iter(self):
        cols = [f"f{i}" for i in range(1, 34)] + ["class"]
        return stream.iter_csv(self.path, target="class", fieldnames=cols)

    @property
    def _repr_content(self):
        return {**super()._repr_content, "Variant": self.variant}
