import abc
import math

from ..base import Branch


class DTBranch(Branch):
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
    def repr_branch(self, index: int, shorten=False):
        """Return a string representation of the test performed in the branch at `index`.

        Parameters
        ----------
        index
            The branch index.
        shorten
            If True, return a shortened version of the performed test.
        """
        pass

class OptionNode(DTBranch):
    """Wrapper node containing all parallel option branches as its children. Doesn't increment depth.
    Options are branches too DEBUG: add reference to paper
    
    Parameters
    ----------
    num_options
        Integer number of option leafs wrapped by this node, used to verify length of stars_per_splitter and 
        splitters.
    stats_per_splitter
        A list of stats. Stats: In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitters
        A list of splitters. Splitter: The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """
    # def __init__(self, num_options, stats_per_splitter, depth, splitters, leaf_model, **kwargs):
    #     assert num_options == len(stats_per_splitter) == len(splitters), \
    #         'Length of stats_per_splitter and of splitters does not match number of leaf options in option node'
    #     # create children (wrapped option nodes of some subclass of DTBranch)
    #     wrapped_options = []
    #     for i in range(num_options):
    #         option = super().__init__(stats_per_splitter[i], depth, splitters[i], leaf_model, **kwargs)
    #         wrapped_options.append(option)
    #     self.wrapped_options = wrapped_options
    def __init__(self, num_options, depth, *children, **kwargs):
        super().__init__(*children)
        self.num_options = num_options
        self.depth = depth

    def next(self, x):
        return self.children

    def walk(self, x, until_leaf=True) -> Iterable[Iterable[Union["Branch", "Leaf", "OptionNode"]]]:
        """Iterate over the nodes of the path induced by x."""
        yield self
        try:
            yield from self.next(x).walk(x, until_leaf)
        except KeyError:
            if until_leaf:
                _, node = self.most_common_path()
                yield node
                yield from node.walk(x, until_leaf)
    
    def traverse(self, x, until_leaf=True) -> Iterable["Leaf"]:
        """Return the leaves corresponding to the given input.

        Alternate option branches are also included.

        Parameters
        ----------
        x
            The input instance.
        until_leaf
            Whether or not branch nodes can be returned in case of missing features or emerging
            feature categories.
        """

        found_nodes = []
        for node in self.walk(x, until_leaf=until_leaf):
            if (
                isinstance(node, AdaBranchRegressor)
                and node._alternate_tree is not None
            ):
                if isinstance(node._alternate_tree, AdaBranchRegressor):
                    found_nodes.append(
                        node._alternate_tree.traverse(x, until_leaf=until_leaf)
                    )
                else:
                    found_nodes.append(node._alternate_tree)

        found_nodes.append(node)  # noqa
        return found_nodes

class NumericBinaryBranch(DTBranch):
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

    def repr_branch(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return f"≤ {round(self.threshold, 4)}"
            return f"> {round(self.threshold, 4)}"
        else:
            if index == 0:
                return f"{self.feature} ≤ {self.threshold}"
            return f"{self.feature} > {self.threshold}"

    @property
    def repr_split(self):
        return f"{self.feature} ≤ {self.threshold}"


class NominalBinaryBranch(DTBranch):
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

    def repr_branch(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return str(self.value)
            else:
                return f"not {self.value}"
        else:
            if index == 0:
                return f"{self.feature} = {self.value}"
            return f"{self.feature} ≠ {self.value}"

    @property
    def repr_split(self):
        return f"{self.feature} {{=, ≠}} {self.value}"


class NumericMultiwayBranch(DTBranch):
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

    def repr_branch(self, index: int, shorten=False):
        lower = self._r_mapping[index] * self.radius
        upper = lower + self.radius

        if shorten:
            return f"[{round(lower, 4)}, {round(upper, 4)})"

        return f"{lower} ≤ {self.feature} < {upper}"

    @property
    def repr_split(self):
        return f"{self.feature} ÷ {self.radius}"


class NominalMultiwayBranch(DTBranch):
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

    def repr_branch(self, index: int, shorten=False):
        feat_val = self._r_mapping[index]

        if shorten:
            return str(feat_val)

        return f"{self.feature} = {feat_val}"

    @property
    def repr_split(self):
        return f"{self.feature} in {set(self._mapping.keys())}"
