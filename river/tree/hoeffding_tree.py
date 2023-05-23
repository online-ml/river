from __future__ import annotations

import collections
import functools
import io
import math
from abc import ABC, abstractmethod

from river import base
from river.utils.norm import normalize_values_in_dict

from .nodes.branch import (
    DTBranch,
    NominalBinaryBranch,
    NominalMultiwayBranch,
    NumericBinaryBranch,
    NumericMultiwayBranch,
)
from .nodes.leaf import HTLeaf
from .utils import calculate_object_size


class HoeffdingTree(ABC):
    """Base class for Hoeffding Decision Trees.

    This is an **abstract class**, so it cannot be used directly. It defines base operations
    and properties that all the Hoeffding decision trees must inherit or implement according to
    their own design.

    Parameters
    ----------
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    binary_split
        If True, only allow binary splits.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.

    """

    def __init__(
        self,
        max_depth: int | None = None,
        binary_split: bool = False,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
    ):
        # Properties common to all the Hoeffding trees
        self._split_criterion: str = ""
        self._leaf_prediction: str = ""

        self.max_depth: float = max_depth if max_depth is not None else math.inf
        self.binary_split: bool = binary_split
        self._max_size: float = max_size
        self._max_byte_size: float = self._max_size * (2**20)  # convert to byte
        self.memory_estimate_period: int = memory_estimate_period
        self.stop_mem_management: bool = stop_mem_management
        self.remove_poor_attrs: bool = remove_poor_attrs
        self.merit_preprune: bool = merit_preprune

        self._root: DTBranch | HTLeaf = None  # type: ignore
        self._n_active_leaves: int = 0
        self._n_inactive_leaves: int = 0
        self._inactive_leaf_size_estimate: float = 0.0
        self._active_leaf_size_estimate: float = 0.0
        self._size_estimate_overhead_fraction: float = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model: float = 0.0

    @staticmethod
    def _hoeffding_bound(range_val, confidence, n):
        r"""Compute the Hoeffding bound, used to decide how many samples are necessary at each
        node.

        Notes
        -----
        The Hoeffding bound is defined as:

        $\\epsilon = \\sqrt{\\frac{R^2\\ln(1/\\delta))}{2n}}$

        where:

        $\\epsilon$: Hoeffding bound.
        $R$: Range of a random variable. For a probability the range is 1, and for an
        information gain the range is log *c*, where *c* is the number of classes.
        $\\delta$: Confidence. 1 minus the desired probability of choosing the correct
        attribute at any given node.
        $n$: Number of samples.

        Parameters
        ----------
        range_val
            Range value.
        confidence
            Confidence of choosing the correct attribute.
        n
            Number of processed samples.
        """
        return math.sqrt((range_val * range_val * math.log(1.0 / confidence)) / (2.0 * n))

    @property
    def max_size(self):
        """Max allowed size tree can reach (in MB)."""
        return self._max_size

    @max_size.setter
    def max_size(self, size):
        self._max_size = size
        self._max_byte_size = self._max_size * (2**20)

    @property
    def height(self) -> int:
        if self._root:
            return self._root.height
        return 0

    @property
    def n_nodes(self):
        if self._root:
            return self._root.n_nodes

    @property
    def n_branches(self):
        if self._root:
            return self._root.n_branches

    @property
    def n_leaves(self):
        if self._root:
            return self._root.n_leaves

    @property
    def n_active_leaves(self):
        return self._n_active_leaves

    @property
    def n_inactive_leaves(self):
        return self._n_inactive_leaves

    @property
    def summary(self):
        """Collect metrics corresponding to the current status of the tree
        in a string buffer.
        """
        summary = {
            "n_nodes": self.n_nodes,
            "n_branches": self.n_branches,
            "n_leaves": self.n_leaves,
            "n_active_leaves": self.n_active_leaves,
            "n_inactive_leaves": self.n_inactive_leaves,
            "height": self.height,
            "total_observed_weight": self._train_weight_seen_by_model,
        }
        return summary

    def to_dataframe(self):
        """Return a representation of the current tree structure organized in a
        `pandas.DataFrame` object.

        In case the tree is empty or it only contains a single node (a leaf), `None` is returned.

        Returns
        -------
        df
            A `pandas.DataFrame` depicting the tree structure.
        """
        if self._root is not None and isinstance(self._root, DTBranch):
            return self._root.to_dataframe()

    def _branch_selector(self, numerical_feature=True, multiway_split=False) -> type[DTBranch]:
        """Create a new split node."""
        if numerical_feature:
            if not multiway_split:
                return NumericBinaryBranch
            else:
                return NumericMultiwayBranch
        else:
            if not multiway_split:
                return NominalBinaryBranch
            else:
                return NominalMultiwayBranch

    @abstractmethod
    def _new_leaf(
        self, initial_stats: dict | None = None, parent: HTLeaf | DTBranch | None = None
    ) -> HTLeaf:
        """Create a new learning node.

        The characteristics of the learning node depends on the tree algorithm.

        Parameters
        ----------
        initial_stats
            Target statistics set from the parent node.
        parent
            Parent node to inherit from.

        Returns
        -------
        A new learning node.
        """

    @property
    def split_criterion(self) -> str:
        """Return a string with the name of the split criterion being used by the tree."""
        return self._split_criterion

    @split_criterion.setter  # type: ignore
    @abstractmethod
    def split_criterion(self, split_criterion):
        """Define the split criterion to be used by the tree."""

    @property
    def leaf_prediction(self) -> str:
        """Return the prediction strategy used by the tree at its leaves."""
        return self._leaf_prediction

    @leaf_prediction.setter  # type: ignore
    @abstractmethod
    def leaf_prediction(self, leaf_prediction):
        """Define the prediction strategy used by the tree in its leaves."""

    def _enforce_size_limit(self):
        """Track the size of the tree and disable/enable nodes if required.

        This memory-management routine shared by all the Hoeffding Trees is based on [^1].

        References
        ----------
        [^1]: Kirkby, R.B., 2007. Improving hoeffding trees (Doctoral dissertation,
        The University of Waikato).
        """
        tree_size = self._size_estimate_overhead_fraction * (
            self._active_leaf_size_estimate
            + self._n_inactive_leaves * self._inactive_leaf_size_estimate
        )
        if self._n_inactive_leaves > 0 or tree_size > self._max_byte_size:
            if self.stop_mem_management:
                self._growth_allowed = False
                return
        leaves = self._find_leaves()
        leaves.sort(key=lambda leaf: leaf.calculate_promise())
        max_active = 0
        while max_active < len(leaves):
            max_active += 1
            if (
                (
                    max_active * self._active_leaf_size_estimate
                    + (len(leaves) - max_active) * self._inactive_leaf_size_estimate
                )
                * self._size_estimate_overhead_fraction
            ) > self._max_byte_size:
                max_active -= 1
                break
        cutoff = len(leaves) - max_active
        for i in range(cutoff):
            if leaves[i].is_active():
                leaves[i].deactivate()
                self._n_inactive_leaves += 1
                self._n_active_leaves -= 1
        for i in range(cutoff, len(leaves)):
            if not leaves[i].is_active() and leaves[i].depth < self.max_depth:
                leaves[i].activate()
                self._n_active_leaves += 1
                self._n_inactive_leaves -= 1

    def _estimate_model_size(self):
        """Calculate the size of the model and trigger tracker function
        if the actual model size exceeds the max size in the configuration.

        This memory-management routine shared by all the Hoeffding Trees is based on [^1].

        References
        ----------
        [^1]: Kirkby, R.B., 2007. Improving hoeffding trees (Doctoral dissertation,
        The University of Waikato).
        """
        leaves = self._find_leaves()
        total_active_size = 0
        total_inactive_size = 0
        for leaf in leaves:
            if leaf.is_active():
                total_active_size += calculate_object_size(leaf)
            else:
                total_inactive_size += calculate_object_size(leaf)
        if total_active_size > 0:
            self._active_leaf_size_estimate = total_active_size / self._n_active_leaves
        if total_inactive_size > 0:
            self._inactive_leaf_size_estimate = total_inactive_size / self._n_inactive_leaves
        actual_model_size = calculate_object_size(self)
        estimated_model_size = (
            self._n_active_leaves * self._active_leaf_size_estimate
            + self._n_inactive_leaves * self._inactive_leaf_size_estimate
        )
        self._size_estimate_overhead_fraction = actual_model_size / estimated_model_size
        if actual_model_size > self._max_byte_size:
            self._enforce_size_limit()

    def _deactivate_all_leaves(self):
        """Deactivate all leaves."""
        leaves = self._find_leaves()
        for leaf in leaves:
            leaf.deactivate()
            self._n_inactive_leaves += 1
            self._n_active_leaves -= 1

    def _find_leaves(self) -> list[HTLeaf]:
        """Find learning nodes in the tree.

        Returns
        -------
        List of learning nodes in the tree.
        """
        return [leaf for leaf in self._root.iter_leaves()]

    # Adapted from creme's original implementation
    def debug_one(self, x: dict) -> str | None:
        """Print an explanation of how `x` is predicted.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
            A representation of the path followed by the tree to predict `x`; `None` if
            the tree is empty.

        Notes
        -----
        Currently, Label Combination Hoeffding Tree Classifier (for multi-label
        classification) is not supported.
        """
        if self._root is None:
            return

        # We'll redirect all the print statement to a buffer, we'll return the content of the
        # buffer at the end
        buffer = io.StringIO()
        _print = functools.partial(print, file=buffer)

        for node in self._root.walk(x, until_leaf=True):
            if isinstance(node, HTLeaf):
                _print(repr(node))
            elif isinstance(node, DTBranch):
                try:
                    child_index = node.branch_no(x)
                except KeyError:
                    child_index, _ = node.most_common_path()

                _print(node.repr_branch(child_index))

        return buffer.getvalue()

    def draw(self, max_depth: int | None = None):
        """Draw the tree using the `graphviz` library.

        Since the tree is drawn without passing incoming samples, classification trees
        will show the majority class in their leaves, whereas regression trees will
        use the target mean.

        Parameters
        ----------
        max_depth
            Only the root will be drawn when set to `0`. Every node will be drawn when
            set to `None`.

        Notes
        -----
        Currently, Label Combination Hoeffding Tree Classifier (for multi-label
        classification) is not supported.

        Examples
        --------
        >>> from river import datasets
        >>> from river import tree
        >>> model = tree.HoeffdingTreeClassifier(
        ...    grace_period=5,
        ...    delta=1e-5,
        ...    split_criterion='gini',
        ...    max_depth=10,
        ...    tau=0.05,
        ... )
        >>> for x, y in datasets.Phishing():
        ...    model = model.learn_one(x, y)
        >>> dot = model.draw()

        .. image:: ../../docs/img/dtree_draw.svg
            :align: center
        """
        try:
            import graphviz
        except ImportError as e:
            raise ValueError("You have to install graphviz to use the draw method.") from e
        counter = 0

        def iterate(node=None):
            if node is None:
                yield None, None, self._root, 0, None
                yield from iterate(self._root)

            nonlocal counter
            parent_no = counter

            if isinstance(node, DTBranch):
                for branch_index, child in enumerate(node.children):
                    counter += 1
                    yield parent_no, node, child, counter, branch_index
                    if isinstance(child, DTBranch):
                        yield from iterate(child)

        if max_depth is None:
            max_depth = -1

        dot = graphviz.Digraph(
            graph_attr={"splines": "ortho", "forcelabels": "true", "overlap": "false"},
            node_attr={
                "shape": "box",
                "penwidth": "1.2",
                "fontname": "trebuchet",
                "fontsize": "11",
                "margin": "0.1,0.0",
            },
            edge_attr={"penwidth": "0.6", "center": "true", "fontsize": "7  "},
        )

        if isinstance(self, base.Classifier):
            n_colors = len(self.classes)  # type: ignore
        else:
            n_colors = 1

        # Pick a color palette which maps classes to colors
        new_color = functools.partial(next, iter(_color_brew(n_colors)))
        palette: collections.defaultdict = collections.defaultdict(new_color)

        for parent_no, parent, child, child_no, branch_index in iterate():
            if child.depth > max_depth and max_depth != -1:
                continue

            if isinstance(child, DTBranch):
                text = f"{child.feature}"  # type: ignore
            else:
                text = f"{repr(child)}\nsamples: {int(child.total_weight)}"

            # Pick a color, the hue depends on the class and the transparency on the distribution
            if isinstance(self, base.Classifier):
                class_proba = normalize_values_in_dict(child.stats, inplace=False)
                mode = max(class_proba, key=class_proba.get)
                p_mode = class_proba[mode]
                try:
                    alpha = (p_mode - 1 / n_colors) / (1 - 1 / n_colors)
                    fillcolor = str(transparency_hex(color=palette[mode], alpha=alpha))
                except ZeroDivisionError:
                    fillcolor = "#FFFFFF"
            else:
                fillcolor = "#FFFFFF"

            dot.node(f"{child_no}", text, fillcolor=fillcolor, style="filled")

            if parent_no is not None:
                dot.edge(
                    f"{parent_no}",
                    f"{child_no}",
                    xlabel=parent.repr_branch(branch_index, shorten=True),
                )

        return dot


# Utility adapted from the original creme's implementation
def _color_brew(n: int) -> list[tuple[int, int, int]]:
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n
        The number of required colors.

    Returns
    -------
        List of n tuples of form (R, G, B) being the components of each color.
    References
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_export.py
    """
    colors = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in [i for i in range(25, 385, int(360 / n))]:
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))

        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]

        # Shift the initial RGB values to match value and store
        colors.append(((int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))))

    return colors


# Utility adapted from the original creme's implementation
def transparency_hex(color: tuple[int, int, int], alpha: float) -> str:
    """Apply alpha coefficient on hexadecimal color."""
    return "#{:02x}{:02x}{:02x}".format(
        *tuple([int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color])
    )
