from __future__ import annotations

import abc

from river import bandit, datasets


class BanditDataset(datasets.base.Dataset):
    """Base class for bandit datasets.

    Parameters
    ----------
    n_features
        Number of features in the dataset.
    n_samples
        Number of samples in the dataset.
    n_classes
        Number of classes in the dataset, only applies to classification datasets.
    n_outputs
        Number of outputs the target is made of, only applies to multi-output datasets.
    sparse
        Whether the dataset is sparse or not.

    """

    def __init__(
        self,
        n_features,
        n_samples=None,
        n_classes=None,
        n_outputs=None,
        sparse=False,
    ):
        super().__init__(
            task="BANDIT",
            n_features=n_features,
            n_samples=n_samples,
            n_classes=n_classes,
            n_outputs=n_outputs,
            sparse=sparse,
        )

    @abc.abstractproperty
    def arms(self) -> list[bandit.base.ArmID]:
        """The list of arms that can be pulled."""

    @property
    def _repr_content(self):
        return {**super()._repr_content, "Arms": f"{len(self.arms):,d}"}
