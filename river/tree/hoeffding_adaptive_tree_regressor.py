from copy import deepcopy

from river.tree import HoeffdingTreeRegressor
from river import base
from river import linear_model

from ._nodes import FoundNode
from ._nodes import LearningNode
from ._nodes import SplitNode
from ._nodes import AdaSplitNodeRegressor
from ._nodes import AdaActiveLearningNodeRegressor
from ._nodes import InactiveLeaf
from ._nodes import InactiveLearningNodeMean
from ._nodes import InactiveLearningNodeModel
from ._nodes import InactiveLearningNodeAdaptive


class HoeffdingAdaptiveTreeRegressor(HoeffdingTreeRegressor):
    """Hoeffding Adaptive Tree regressor (HATR).

    This class implements a regression version of the Hoeffding Adaptive Tree Classifier. Hence,
    it also uses an ADWIN concept-drift detector instance at each decision node to monitor
    possible changes in the data distribution. If a drift is detected in a node, an alternate
    tree begins to be induced in the background. When enough information is gathered, HATR
    swaps the node where the change was detected by its alternate tree.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        | Prediction mechanism used at leafs.
        | 'mean' - Target mean
        | 'model' - Uses the model defined in `leaf_model`
        | 'adaptive' - Chooses between 'mean' and 'model' dynamically
    leaf_model
        The regression model used to provide responses if `leaf_prediction='model'`. If not
        provided an instance of `river.linear_model.LinearRegression` with the default
        hyperparameters is used.
    model_selector_decay
        The exponential decaying factor applied to the learning models' squared errors, that
        are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer
        to `1`, the more importance is going to be given to past observations. On the other hand,
        if its value approaches `0`, the recent observed errors are going to have more influence
        on the final decision.
    nominal_attributes
        List of Nominal attributes. If empty, then assume that all numeric attributes should
        be treated as continuous.
    bootstrap_sampling
        If True, perform bootstrap sampling in the leaf nodes.
    drift_window_threshold
        Minimum number of examples an alternate tree must observe before being considered as a
        potential replacement to the current one.
    adwin_confidence
        The delta parameter used in the nodes' ADWIN drift detectors.
    seed
       If int, `seed` is the seed used by the random number generator;
       If RandomState instance, `seed` is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`. Only used when `bootstrap_sampling=True` to direct the
       bootstrap sampling.

    Notes
    -----
    The Hoeffding Adaptive Tree [^1] uses ADWIN [^2] to monitor performance of branches on the tree
    and to replace them with new branches when their accuracy decreases if the new branches are
    more accurate.

    The bootstrap sampling strategy is an improvement over the original Hoeffding Adaptive Tree
    algorithm. It is enabled by default since, in general, it results in better performance.

    To cope with ADWIN's requirements of bounded input data, HATR uses a novel error normalization
    strategy based on the empiral rule of Gaussian distributions. We assume the deviations
    of the predictions from the expected values follow a normal distribution. Hence, we subject
    these errors to a min-max normalization assuming that most of the data lies in the
    $\\left[-3\\sigma, 3\\sigma\\right]$ range. These normalized errors are passed to the ADWIN
    instances. This is the same strategy used by Adaptive Random Forest Regressor.

    References
    ----------
    [^1]: Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data streams."
    In International Symposium on Intelligent Data Analysis, pp. 249-260. Springer, Berlin,
    Heidelberg, 2009.
    [^2]: Bifet, Albert, and Ricard Gavaldà. "Learning from time-changing data with adaptive
    windowing." In Proceedings of the 2007 SIAM international conference on data mining,
    pp. 443-448. Society for Industrial and Applied Mathematics, 2007.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     tree.HoeffdingAdaptiveTreeRegressor(
    ...         grace_period=50,
    ...         leaf_prediction='adaptive',
    ...         model_selector_decay=0.3,
    ...         seed=0
    ...     )
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.78838
    """

    def __init__(self,
                 grace_period: int = 200,
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'model',
                 leaf_model: base.Regressor = None,
                 model_selector_decay: float = 0.95,
                 nominal_attributes: list = None,
                 bootstrap_sampling: bool = True,
                 drift_window_threshold: int = 300,
                 adwin_confidence: float = 0.002,
                 seed=None,
                 **kwargs):

        super().__init__(grace_period=grace_period,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         leaf_prediction=leaf_prediction,
                         leaf_model=leaf_model,
                         model_selector_decay=model_selector_decay,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self._n_alternate_trees = 0
        self._n_pruned_alternate_trees = 0
        self._n_switch_alternate_trees = 0

        self.bootstrap_sampling = bootstrap_sampling
        self.drift_window_threshold = drift_window_threshold
        self.adwin_confidence = adwin_confidence
        self.seed = seed

    def reset(self):
        super().reset()
        self._n_alternate_trees = 0
        self._n_pruned_alternate_trees = 0
        self._n_switch_alternate_trees = 0

    def learn_one(self, x, y, *, sample_weight=1.):
        self._train_weight_seen_by_model += sample_weight

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._n_active_leaves = 1
        if isinstance(self._tree_root, InactiveLeaf):
            self._tree_root.learn_one(x, y, weight=sample_weight, tree=self)
        else:
            self._tree_root.learn_one(x, y, sample_weight, self, None, -1)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    def predict_one(self, x):
        if self._tree_root is not None:
            if isinstance(self._tree_root, InactiveLeaf):
                found_nodes = [self._tree_root.filter_instance_to_leaf(x, None, -1)]
            else:
                found_nodes = self._filter_instance_to_leaves(x, None, -1)

            pred = 0.
            for fn in found_nodes:
                # parent_branch == -999 means that the node is the root of an alternate tree.
                # In other words, the alternate tree is a single leaf. It is probably not accurate
                # enough to be used to predict, so skip it
                if fn.parent_branch != -999:
                    leaf_node = fn.node
                    if leaf_node is None:
                        leaf_node = fn.parent
                    # Option Tree prediction (of sorts): combine the response of all leaves reached
                    # by the instance
                    pred += leaf_node.predict_one(x, tree=self)
            # Mean prediction among the reached leaves
            return pred / len(found_nodes)
        else:
            return 0.

    def _filter_instance_to_leaves(self, x, split_parent, parent_branch):
        nodes = []
        self._tree_root.filter_instance_to_leaves(x, split_parent, parent_branch, nodes)
        return nodes

    def _new_learning_node(self, initial_stats=None, parent=None, is_active=True):
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """
        if parent is not None:
            depth = parent.depth + 1
            # Leverage ancestor's learning models
            if not isinstance(parent, AdaSplitNodeRegressor):
                leaf_model = deepcopy(parent._leaf_model)
            else:  # Corner case where an alternate tree is created
                # TODO: change to appropriate 'clone' method
                leaf_model = self.leaf_model.__class__(**self.leaf_model._get_params())
        else:
            depth = 0
            # TODO: change to appropriate 'clone' method
            leaf_model = self.leaf_model.__class__(**self.leaf_model._get_params())

        if is_active:
            new_ada_leaf = AdaActiveLearningNodeRegressor(
                initial_stats=initial_stats, depth=depth, leaf_model=leaf_model,
                adwin_delta=self.adwin_confidence, seed=self.seed)

            if parent is not None and not isinstance(parent, SplitNode):
                new_ada_leaf._fmse_mean = parent._fmse_mean
                new_ada_leaf._fmse_model = parent._fmse_model

            return new_ada_leaf
        else:
            if self.leaf_prediction == self._TARGET_MEAN:
                return InactiveLearningNodeMean(initial_stats, depth)
            elif self.leaf_prediction == self._MODEL:
                return InactiveLearningNodeModel(initial_stats, depth, leaf_model)
            else:  # adaptive learning node
                new_adaptive = InactiveLearningNodeAdaptive(initial_stats, depth, leaf_model)
                if parent is not None and not isinstance(parent, SplitNode):
                    self._fmse_mean = parent._fmse_mean
                    self._fmse_model = parent._fmse_model

                return new_adaptive

    def _new_split_node(self, split_test, target_stats=None, depth=0):
        return AdaSplitNodeRegressor(
            split_test=split_test, stats=target_stats, depth=depth,
            adwin_delta=self.adwin_confidence, seed=self.seed)

    # Override river.tree.DecisionTree to include alternate trees
    def __find_learning_nodes(self, node, parent, parent_branch, found):
        if node is not None:
            if isinstance(node, LearningNode):
                found.append(FoundNode(node, parent, parent_branch))
            if isinstance(node, SplitNode):
                split_node = node
                for i in range(split_node.n_children):
                    self.__find_learning_nodes(
                        split_node.get_child(i), split_node, i, found)
                if split_node._alternate_tree is not None:
                    self.__find_learning_nodes(
                        split_node._alternate_tree, split_node, -999, found)

    # Override river.tree.DecisionTree to include alternate trees
    def _deactivate_leaf(self, to_deactivate, parent, parent_branch):
        new_leaf = self._new_learning_node(to_deactivate.stats, parent=to_deactivate,
                                           is_active=False)
        new_leaf.depth -= 1  # To ensure we do not skip a tree level
        if parent is None:
            self._tree_root = new_leaf
        else:
            # Corner case where a leaf is the alternate tree
            if parent_branch == -999:
                parent._alternate_tree = new_leaf

            else:
                parent.set_child(parent_branch, new_leaf)
        self._n_active_leaves -= 1
        self._n_inactive_leaves += 1

    # Override river.tree.DecisionTree to include alternate trees
    def _activate_leaf(self, to_activate, parent, parent_branch):
        new_leaf = self._new_learning_node(to_activate.stats, parent=to_activate)
        new_leaf.depth -= 1  # To ensure we do not skip a tree level
        if parent is None:
            self._tree_root = new_leaf
        else:
            # Corner case where a leaf is the alternate tree
            if parent_branch == -999:
                parent._alternate_tree = new_leaf
            else:
                parent.set_child(parent_branch, new_leaf)
        self._n_active_leaves += 1
        self._n_inactive_leaves -= 1
