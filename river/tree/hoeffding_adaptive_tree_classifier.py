import typing

from river.utils.skmultiflow_utils import add_dict_values, normalize_values_in_dict

from .hoeffding_tree_classifier import HoeffdingTreeClassifier
from .nodes.branch import HTBranch
from .nodes.hatc_nodes import (
    AdaBranchClassifier,
    AdaLeafClassifier,
    AdaNomBinaryBranchClass,
    AdaNomMultiwayBranchClass,
    AdaNumBinaryBranchClass,
    AdaNumMultiwayBranchClass,
)
from .splitter import Splitter


class HoeffdingAdaptiveTreeClassifier(HoeffdingTreeClassifier):
    """Hoeffding Adaptive Tree classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes. If empty, then assume that all numeric attributes should
        be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    bootstrap_sampling
        If True, perform bootstrap sampling in the leaf nodes.
    drift_window_threshold
        Minimum number of examples an alternate tree must observe before being considered as a
        potential replacement to the current one.
    adwin_confidence
        The delta parameter used in the nodes' ADWIN drift detectors.
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
    seed
       If int, `seed` is the seed used by the random number generator;</br>
       If RandomState instance, `seed` is the random number generator;</br>
       If None, the random number generator is the RandomState instance used
       by `np.random`. Only used when `bootstrap_sampling=True` to direct the
       bootstrap sampling.</br>


    Notes
    -----
    The Hoeffding Adaptive Tree [^1] uses ADWIN [^2] to monitor performance of branches on the tree
    and to replace them with new branches when their accuracy decreases if the new branches are
    more accurate.

    The bootstrap sampling strategy is an improvement over the original Hoeffding Adaptive Tree
    algorithm. It is enabled by default since, in general, it results in better performance.

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
    >>> from river import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
    ...                                drift_stream=synth.SEA(seed=42, variant=1),
    ...                                seed=1, position=500, width=50)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.HoeffdingAdaptiveTreeClassifier(
    ...     grace_period=100,
    ...     split_confidence=1e-5,
    ...     leaf_prediction='nb',
    ...     nb_threshold=10,
    ...     seed=0
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 91.09%

    """

    # =============================================
    # == Hoeffding Adaptive Tree implementation ===
    # =============================================

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        bootstrap_sampling: bool = True,
        drift_window_threshold: int = 300,
        adwin_confidence: float = 0.002,
        binary_split: bool = False,
        max_size: int = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed=None,
    ):

        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self._n_alternate_trees = 0
        self._n_pruned_alternate_trees = 0
        self._n_switch_alternate_trees = 0

        self.bootstrap_sampling = bootstrap_sampling
        self.drift_window_threshold = drift_window_threshold
        self.adwin_confidence = adwin_confidence
        self.seed = seed

    @property
    def n_alternate_trees(self):
        return self._n_alternate_trees

    @property
    def n_pruned_alternate_trees(self):
        return self._n_pruned_alternate_trees

    @property
    def n_switch_alternate_trees(self):
        return self._n_switch_alternate_trees

    @property
    def summary(self):
        summ = super().summary
        summ.update(
            {
                "n_alternate_trees": self.n_alternate_trees,
                "n_pruned_alternate_trees": self.n_pruned_alternate_trees,
                "n_switch_alternate_trees": self.n_switch_alternate_trees,
            }
        )
        return summ

    def learn_one(self, x, y, *, sample_weight=1.0):
        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        self._root.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    # Override HoeffdingTreeClassifier
    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in self.classes}
        if self._root is not None:
            found_nodes = [self._root]
            if isinstance(self._root, HTBranch):
                found_nodes = self._root.traverse(x, until_leaf=True)
            for leaf in found_nodes:
                dist = leaf.prediction(x, tree=self)
                # Option Tree prediction (of sorts): combine the response of all leaves reached
                # by the instance
                proba = add_dict_values(proba, dist, inplace=True)
            proba = normalize_values_in_dict(proba)

        return proba

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}

        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        return AdaLeafClassifier(
            stats=initial_stats,
            depth=depth,
            splitter=self.splitter,
            adwin_delta=self.adwin_confidence,
            seed=self.seed,
        )

    def _branch_selector(
        self, numerical_feature=True, multiway_split=False
    ) -> typing.Type[AdaBranchClassifier]:
        """Create a new split node."""
        if numerical_feature:
            if not multiway_split:
                return AdaNumBinaryBranchClass
            else:
                return AdaNumMultiwayBranchClass
        else:
            if not multiway_split:
                return AdaNomBinaryBranchClass
            else:
                return AdaNomMultiwayBranchClass

    @classmethod
    def _unit_test_params(cls):
        return {"seed": 1}
