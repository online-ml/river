import abc
from river import bandit
from river import datasets


class BanditDataset(datasets.base.Dataset):
    @abc.abstractproperty
    def arms(self) -> list[bandit.base.ArmID]:
        """The list of arms that can be pulled."""

    @property
    def _repr_content(self):
        return {**super()._repr_content, "Arms": f"{len(self.arms):,d}"}
