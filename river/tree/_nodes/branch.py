import abc
import math

from ..base import Branch


class HTBranch(Branch):
    def __init__(self, *children):
        super().__init__(*children)

    @property
    def total_weight(self):
        return sum(child.total_weight for child in filter(None, self.children))

    @property
    @abc.abstractmethod
    def max_branches(self):
        pass

    @abc.abstractmethod
    def most_common_path(self):
        pass


class NumericBinaryBranch(HTBranch):
    def __init__(self, feature, threshold, depth, left, right, **attributes):
        super().__init__(left, right)
        self.feature = feature
        self.threshold = threshold
        self.depth = depth
        self.__dict__.update(attributes)

    def next(self, x):
        left, right = self.children

        if x[self.feature] <= self.threshold:
            return left
        return right

    @property
    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return right
        return left


class NominalBinaryBranch(HTBranch):
    def __init__(self, feature, value, depth, left, right, **attributes):
        super().__init__(left, right)
        self.feature = feature
        self.value = value
        self.depth = depth
        self.__dict__.update(attributes)

    def next(self, x):
        left, right = self.children

        if x[self.feature] == self.value:
            return left
        return right

    @property
    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return right
        return left


class NumericMultiwayBranch(HTBranch):
    def __init__(self, feature, radius, depth, slot_ids, *children, **attributes):
        super().__init__(*children)
        # The number of branches can increase in runtime
        self.children = list(self.children)

        self.feature = feature
        self.radius = radius
        self.depth = depth

        # Controls the branch mapping
        self._mapping = {s: i for i, s in enumerate(slot_ids)}
        self._r_mapping = {i: s for s, i in self._mapping.items()}

        self.__dict__.update(attributes)

    def next(self, x):
        slot = math.floor(x[self.feature] / self.radius)
        pos = self._mapping[slot]

        return self.children[pos]

    @property
    def max_branches(self):
        return -1

    def most_common_path(self):
        # Get the most traversed path
        pos = max(
            range(len(self.children)), key=lambda i: self.children[i].total_weight
        )

        return self.children[pos]

    def add_child(self, feature_val, child):
        slot = math.floor(feature_val / self.radius)

        self._mapping[slot] = len(self.children)
        self._r_mapping[len(self.children)] = slot
        self.children.append(child)


class NominalMultiwayBranch(HTBranch):
    def __init__(self, feature, feature_values, depth, *children, **attributes):
        super().__init__(*children)
        # The number of branches can increase in runtime
        self.children = list(self.children)

        self.feature = feature
        self.depth = depth

        # Controls the branch mapping
        self._mapping = {feat_v: i for i, feat_v in enumerate(feature_values)}
        self._r_mapping = {i: feat_v for feat_v, i in self._mapping.items()}

        self.__dict__.update(attributes)

    def next(self, x):
        pos = self._mapping[x[self.feature]]

        return self.children[pos]

    @property
    def max_branches(self):
        return -1

    def most_common_path(self):
        # Get the most traversed path
        pos = max(
            range(len(self.children)), key=lambda i: self.children[i].total_weight
        )

        return self.children[pos]

    def add_child(self, feature_val, child):
        self._mapping[feature_val] = len(self.children)
        self._r_mapping[len(self.children)] = feature_val
        self.children.append(child)
