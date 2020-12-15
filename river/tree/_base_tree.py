from abc import ABC, abstractmethod
import collections
import functools
import io
import math
import typing

from river import base
from river.utils.skmultiflow_utils import calculate_object_size
from river.utils.skmultiflow_utils import normalize_values_in_dict
from river.utils.skmultiflow_utils import round_sig_fig

from ._nodes import Node
from ._nodes import LearningNode
from ._nodes import SplitNode
from ._nodes import FoundNode
from ._attribute_test import InstanceConditionalTest

try:
    import graphviz

    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False


class BaseHoeffdingTree(ABC):
    """Base class for Hoeffding Decision Trees.

    This is an **abstract class**, so it cannot be used directly. It defines base operations
    and properties that all the decision trees must inherit or implement according to
    their own design.

    All the extended classes inherit the following functionality:

    * Set the maximum tree depth allowed (`max_depth`).

    * Handle *Active* and *Inactive* nodes: Active learning nodes update their own
    internal state to improve predictions and monitor input features to perform split
    attempts. Inactive learning nodes do not update their internal state and only keep the
    predictors; they are used to save memory in the tree (`max_size`).

    *  Enable/disable memory management.

    * Define strategies to sort leaves according to how likely they are going to be split.
    This enables deactivating non-promising leaves to save memory.

    * Disabling ‘poor’ attributes to save memory and speed up tree construction.
    A poor attribute is an input feature whose split merit is much smaller than the current
    best candidate. Once a feature is disabled, the tree stops saving statistics necessary
    to split such a feature.

    * Define properties to access leaf prediction strategies, split criteria, and other
    relevant characteristics.

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

    Notes
    -----
    Hoeffding trees might face situations where an input feature previously used to make
    a split decision is missing in an incoming sample. In this case, the trees follow the
    conventions:

    - *Learning:* choose the subtree branch most traversed so far to pass the instance on.</br>
    - *Predicting:* Use the last "reachable" decision node to provide responses.

    """

    def __init__(
        self,
        max_depth: int = None,
        binary_split: bool = False,
        max_size: int = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
    ):
        # Properties common to all the Hoeffding trees
        self._split_criterion: str
        self._leaf_prediction: str

        self.max_depth: float = max_depth if max_depth is not None else math.inf
        self.binary_split: bool = binary_split
        self._max_size: float = max_size
        self._max_byte_size: float = self._max_size * (2 ** 20)  # convert to byte
        self.memory_estimate_period: int = memory_estimate_period
        self.stop_mem_management: bool = stop_mem_management
        self.remove_poor_attrs: bool = remove_poor_attrs
        self.merit_preprune: bool = merit_preprune

        self._tree_root: typing.Union[Node, None] = None
        self._n_decision_nodes: int = 0
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
        self._max_byte_size = self._max_size * (2 ** 20)

    @property
    def model_measurements(self):
        """Collect metrics corresponding to the current status of the tree
        in a string buffer.
        """
        measurements = {
            "Tree size (nodes)": (
                self._n_decision_nodes + self._n_active_leaves + self._n_inactive_leaves
            ),
            "Tree size (leaves)": self._n_active_leaves + self._n_inactive_leaves,
            "Active learning nodes": self._n_active_leaves,
            "Tree depth": self.depth,
            "Active leaf byte size estimate": self._active_leaf_size_estimate,
            "Inactive leaf byte size estimate": self._inactive_leaf_size_estimate,
            "Byte size estimate overhead": self._size_estimate_overhead_fraction,
        }
        return measurements

    def model_description(self):
        """Walk the tree and return its structure in a buffer.

        Returns
        -------
        The description of the model.

        """
        if self._tree_root is not None:
            buffer = [""]
            description = ""
            self._tree_root.describe_subtree(self, buffer, 0)
            for line in range(len(buffer)):
                description += buffer[line]
            return description

    def _new_split_node(
        self,
        split_test: InstanceConditionalTest,
        target_stats: dict = None,
        depth: int = 0,
    ) -> SplitNode:
        """Create a new split node."""
        return SplitNode(split_test, target_stats, depth)

    @abstractmethod
    def _new_learning_node(self, initial_stats: dict = None, parent: Node = None) -> LearningNode:
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
    def depth(self) -> int:
        """The depth of the tree."""
        if isinstance(self._tree_root, Node):
            return self._tree_root.subtree_depth()
        return 0

    @property
    def split_criterion(self) -> str:
        """Return a string with the name of the split criterion being used by the tree. """
        return self._split_criterion

    @split_criterion.setter
    @abstractmethod
    def split_criterion(self, split_criterion):
        """Define the split criterion to be used by the tree. """

    @property
    def leaf_prediction(self) -> str:
        """Return the prediction strategy used by the tree at its leaves. """
        return self._leaf_prediction

    @leaf_prediction.setter
    @abstractmethod
    def leaf_prediction(self, leaf_prediction):
        """Define the prediction strategy used by the tree in its leaves."""

    def _enforce_size_limit(self):
        """Track the size of the tree and disable/enable nodes if required."""
        tree_size = self._size_estimate_overhead_fraction * (
            self._active_leaf_size_estimate
            + self._n_inactive_leaves * self._inactive_leaf_size_estimate
        )
        if self._n_inactive_leaves > 0 or tree_size > self._max_byte_size:
            if self.stop_mem_management:
                self._growth_allowed = False
                return
        learning_nodes = self._find_learning_nodes()
        learning_nodes.sort(key=lambda n: n.node.calculate_promise())
        max_active = 0
        while max_active < len(learning_nodes):
            max_active += 1
            if (
                (
                    max_active * self._active_leaf_size_estimate
                    + (len(learning_nodes) - max_active) * self._inactive_leaf_size_estimate
                )
                * self._size_estimate_overhead_fraction
            ) > self._max_byte_size:
                max_active -= 1
                break
        cutoff = len(learning_nodes) - max_active
        for i in range(cutoff):
            if learning_nodes[i].node.is_active():
                learning_nodes[i].node.deactivate()
                self._n_inactive_leaves += 1
                self._n_active_leaves -= 1
        for i in range(cutoff, len(learning_nodes)):
            if (
                not learning_nodes[i].node.is_active()
                and learning_nodes[i].node.depth < self.max_depth
            ):
                learning_nodes[i].node.activate()
                self._n_active_leaves += 1
                self._n_inactive_leaves -= 1

    def _estimate_model_size(self):
        """Calculate the size of the model and trigger tracker function
        if the actual model size exceeds the max size in the configuration."""
        learning_nodes = self._find_learning_nodes()
        total_active_size = 0
        total_inactive_size = 0
        for found_node in learning_nodes:
            if not found_node.node.is_leaf():  # Safety check for non-trivial tree structures
                continue
            if found_node.node.is_active():
                total_active_size += calculate_object_size(found_node.node)
            else:
                total_inactive_size += calculate_object_size(found_node.node)
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
        """Deactivate all leaves. """
        learning_nodes = self._find_learning_nodes()
        for cur_node in learning_nodes:
            cur_node.node.deactivate()
            self._n_inactive_leaves += 1
            self._n_active_leaves -= 1

    def _find_learning_nodes(self) -> typing.List[FoundNode]:
        """Find learning nodes in the tree.

        Returns
        -------
        List of learning nodes in the tree.
        """
        found_list: typing.List[FoundNode] = []
        self.__find_learning_nodes(self._tree_root, None, -1, found_list)
        return found_list

    def __find_learning_nodes(self, node, parent, parent_branch, found):
        """Find learning nodes in the tree from a given node.

        Parameters
        ----------
        node
            The node to start the search.
        parent
            The node's parent.
        parent_branch
            Parent node's branch.
        found
            A list of found nodes.

        Returns
        -------
        List of learning nodes.
        """
        if node is not None:
            if node.is_leaf():
                found.append(FoundNode(node, parent, parent_branch))
            else:
                split_node = node
                for i in range(split_node.n_children):
                    self.__find_learning_nodes(split_node.get_child(i), split_node, i, found)

    # Adapted from creme's original implementation
    def debug_one(self, x: dict) -> typing.Union[str, None]:
        """Print an explanation of how `x` is predicted.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
            A representation of the path followed by the tree to predict `x`; `None` if
            the tree is empty.
        """
        if self._tree_root is None:
            return

        # We'll redirect all the print statement to a buffer, we'll return the content of the
        # buffer at the end
        buffer = io.StringIO()
        _print = functools.partial(print, file=buffer)

        for node in self._tree_root.path(x):
            if node.is_leaf():
                pred = node.leaf_prediction(x, tree=self)
                if isinstance(self, base.Classifier):
                    class_val = max(pred, key=pred.get)
                    _print(f"Class {class_val} | {pred}")
                else:
                    # Multi-target regression case
                    if isinstance(self, base.MultiOutputMixin):
                        _print("Predictions:\n{")
                        for i, (t, var) in enumerate(pred.items()):
                            _print(f"\t{t}: {pred[t]} | {node.stats[t].mean} | {node.stats[t]}")
                        _print("}")
                    else:  # Single-target regression
                        _print(f"Prediction {pred} | {node.stats.mean} | {node.stats}")
                break
            else:
                child_index = node.split_test.branch_for_instance(x)

                if child_index >= 0:
                    _print(node.split_test.describe_condition_for_branch(child_index))
                else:  # Corner case where an emerging nominal feature value arrives
                    _print("Decision node reached as final destination")
                    pred = node.stats
                    if isinstance(self, base.Classifier):
                        class_val = max(pred, key=pred.get)
                        pred = normalize_values_in_dict(pred, inplace=False)
                        _print(f"Class {class_val} | {pred}")
                    else:
                        # Multi-target regression case
                        if isinstance(self, base.MultiOutputMixin):
                            _print("Predictions:\n{")
                            for i, (t, var) in enumerate(pred.items()):
                                _print(f"\t{t}: {pred[t].mean.get()} | {pred[t]}")
                            _print("}")
                        else:  # Single-target regression
                            _print(f"Prediction {pred} | {node.stats.mean} | {node.stats}")

        return buffer.getvalue()

    def draw(self, max_depth: int = None):
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
        ...    split_confidence=1e-5,
        ...    split_criterion='gini',
        ...    max_depth=10,
        ...    tie_threshold=0.05,
        ... )
        >>> for x, y in datasets.Phishing():
        ...    model = model.learn_one(x, y)
        >>> dot = model.draw()

        .. image:: ../../docs/img/dtree_draw.svg
            :align: center
        """

        def node_prediction(node):
            if isinstance(self, base.Classifier):
                pred = node.stats
                text = str(max(pred, key=pred.get))
                sum_votes = sum(pred.values())
                if sum_votes > 0:
                    pred = normalize_values_in_dict(pred, factor=sum_votes, inplace=False)
                    probas = "\n".join(
                        [f"P({c}) = {round_sig_fig(proba)}" for c, proba in pred.items()]
                    )
                    text = f"{text}\n{probas}"
                return text
            elif isinstance(self, base.Regressor):
                # Multi-target regression
                if isinstance(self, base.MultiOutputMixin):
                    return " | ".join(
                        [f"{t} = {round_sig_fig(s.mean.get())}" for t, s in node.stats.items()]
                    )
                else:  # vanilla single-target regression
                    pred = node.stats.mean.get()
                    return f"{round_sig_fig(pred)}"

        if max_depth is None:
            max_depth = math.inf

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
            n_colors = len(self.classes)  # noqa
        else:
            n_colors = 1

        # Pick a color palette which maps classes to colors
        new_color = functools.partial(next, iter(_color_brew(n_colors)))
        palette = collections.defaultdict(new_color)

        for (
            parent_no,
            child_no,
            parent,
            child,
            branch_id,
        ) in self._tree_root.iter_edges():

            if child.depth > max_depth:
                continue

            if not child.is_leaf():
                text = f"{child.split_test.attrs_test_depends_on()[0]}"

                if child.depth == max_depth:
                    text = f"{text}\n{node_prediction(child)}"
            else:
                text = f"{node_prediction(child)}\nsamples: {int(child.total_weight)}"

            # Pick a color, the hue depends on the class and the transparency on the distribution
            if isinstance(self, base.Classifier):
                class_proba = {c: 0 for c in self.classes}  # noqa
                class_proba.update(normalize_values_in_dict(child.stats, inplace=False))
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
                    xlabel=parent.split_test.describe_condition_for_branch(branch_id, shorten=True),
                )

        return dot


# Utility adapted from the original creme's implementation
def _color_brew(n: int) -> typing.List[typing.Tuple[int, int, int]]:
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
def transparency_hex(color: typing.Tuple[int, int, int], alpha: float) -> str:
    """Apply alpha coefficient on hexadecimal color."""
    return "#%02x%02x%02x" % tuple([int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color])
