import abc
import math

from ..base import Branch


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

    @abc.abstractmethod
    def max_branches(self):
        pass

    @abc.abstractmethod
    def repr_split(self, index: int, shorten=False):
        """Return a string representation of the test performed in the branch at `index`.

        Parameters
        ----------
        index
            The branch index.
        shorten
            If True, return a shortened version of the performed test.
        """
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

    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return 1, right
        return 0, left

    def repr_split(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return f"≤ {round(self.threshold, 4)}"
            return f"> {round(self.threshold, 4)}"
        else:
            if index == 0:
                return f"{self.feature} ≤ {self.threshold}"
            return f"{self.feature} > {self.threshold}"


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

    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return 1, right
        return 0, left

    def repr_split(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return str(self.value)
            else:
                return f"not {self.value}"
        else:
            if index == 0:
                return f"{self.feature} = {self.value}"
            return f"{self.feature} ≠ {self.value}"


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

    def repr_split(self, index: int, shorten=False):
        lower = self._r_mapping[index] * self.radius
        upper = lower + self.radius

        if shorten:
            return f"[{round(lower, 4)}, {round(upper, 4)})"

        return f"{lower} ≤ {self.feature} < {upper}"


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

    def repr_split(self, index: int, shorten=False):
        feat_val = self._r_mapping[index]

        if shorten:
            return str(feat_val)

        return f"{self.feature} = {feat_val}"
