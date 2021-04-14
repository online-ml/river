import abc
import math
import typing

from ..base import Branch
from .leaf import HTLeaf


class HTBranch(Branch):
    def __init__(self, stats, *children, **attributes):
        super().__init__(*children)
        # The number of branches can increase in runtime
        self.children = list(self.children)

        self.stats = stats
        self.__dict__.update(attributes)

    @property
    def total_weight(self):
        return sum(child.total_weight for child in filter(None, self.children))

    @abc.abstractmethod
    def branch_no(self, x):
        pass

    def next(self, x):
        return self.children[self.branch_no(x)]

    @property
    @abc.abstractmethod
    def max_branches(self):
        pass

    @abc.abstractmethod
    def most_common_path(self) -> typing.Tuple[int, typing.Union["HTLeaf", "HTBranch"]]:
        pass


class NumericBinaryBranch(HTBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        super().__init__(stats, left, right, **attributes)
        self.feature = feature
        self.threshold = threshold
        self.depth = depth

    def branch_no(self, x):
        if x[self.feature] <= self.threshold:
            return 0
        return 1

    @property
    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return 1, right
        return 0, left


class NominalBinaryBranch(HTBranch):
    def __init__(self, stats, feature, value, depth, left, right, **attributes):
        super().__init__(stats, left, right, **attributes)
        self.feature = feature
        self.value = value
        self.depth = depth

    def branch_no(self, x):
        if x[self.feature] == self.value:
            return 0
        return 1

    @property
    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return 1, right
        return 0, left


class NumericMultiwayBranch(HTBranch):
    def __init__(
        self, stats, feature, radius_and_slots, depth, *children, **attributes
    ):
        super().__init__(stats, *children, **attributes)

        self.feature = feature
        self.radius, slot_ids = radius_and_slots
        self.depth = depth

        # Controls the branch mapping
        self._mapping = {s: i for i, s in enumerate(slot_ids)}
        self._r_mapping = {i: s for s, i in self._mapping.items()}

    def branch_no(self, x):
        slot = math.floor(x[self.feature] / self.radius)

        return self._mapping[slot]

    @property
    def max_branches(self):
        return -1

    def most_common_path(self):
        # Get the most traversed path
        pos = max(
            range(len(self.children)), key=lambda i: self.children[i].total_weight
        )

        return pos, self.children[pos]

    def add_child(self, feature_val, child):
        slot = math.floor(feature_val / self.radius)

        self._mapping[slot] = len(self.children)
        self._r_mapping[len(self.children)] = slot
        self.children.append(child)


class NominalMultiwayBranch(HTBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **attributes):
        super().__init__(stats, *children, **attributes)
        self.feature = feature
        self.depth = depth

        # Controls the branch mapping
        self._mapping = {feat_v: i for i, feat_v in enumerate(feature_values)}
        self._r_mapping = {i: feat_v for feat_v, i in self._mapping.items()}

    def branch_no(self, x):
        return self._mapping[x[self.feature]]

    @property
    def max_branches(self):
        return -1

    def most_common_path(self):
        # Get the most traversed path
        pos = max(
            range(len(self.children)), key=lambda i: self.children[i].total_weight
        )

        return pos, self.children[pos]

    def add_child(self, feature_val, child):
        self._mapping[feature_val] = len(self.children)
        self._r_mapping[len(self.children)] = feature_val
        self.children.append(child)
