import numpy as np

from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import add_dict_values

from ._nodes import InactiveLeaf
from ._nodes import InactiveLearningNodeMC
from ._nodes import AdaLearningNode
from ._nodes import AdaSplitNode

import warnings


def HAT(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
        split_criterion='info_gain', split_confidence=0.0000001, tie_threshold=0.05,
        binary_split=False, stop_mem_management=False, remove_poor_atts=False, no_preprune=False,
        leaf_prediction='nba', nb_threshold=0, nominal_attributes=None, bootstrap_sampling=True,
        random_state=None):     # pragma: no cover
    warnings.warn("'HAT' has been renamed to 'HoeffdingAdaptiveTreeClassifier' in v0.5.0.\n"
                  "The old name will be removed in v0.7.0", category=FutureWarning)
    return HoeffdingAdaptiveTreeClassifier(max_byte_size=max_byte_size,
                                           memory_estimate_period=memory_estimate_period,
                                           grace_period=grace_period,
                                           split_criterion=split_criterion,
                                           split_confidence=split_confidence,
                                           tie_threshold=tie_threshold,
                                           binary_split=binary_split,
                                           stop_mem_management=stop_mem_management,
                                           remove_poor_atts=remove_poor_atts,
                                           no_preprune=no_preprune,
                                           leaf_prediction=leaf_prediction,
                                           nb_threshold=nb_threshold,
                                           nominal_attributes=nominal_attributes,
                                           bootstrap_sampling=bootstrap_sampling,
                                           random_state=random_state)


class HoeffdingAdaptiveTreeClassifier(HoeffdingTreeClassifier):
    """ Hoeffding Adaptive Tree classifier.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.

    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.

    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.

    split_criterion: string (default='info_gain')
        Split criterion to use.

        - 'gini' - Gini
        - 'info_gain' - Information Gain

    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.

    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.

    binary_split: boolean (default=False)
        If True, only allow binary splits.

    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.

    remove_poor_atts: boolean (default=False)
        If True, disable poor attributes.

    no_preprune: boolean (default=False)
        If True, disable pre-pruning.

    leaf_prediction: string (default='nba')
        Prediction mechanism used at leafs.

        - 'mc' - Majority Class
        - 'nb' - Naive Bayes
        - 'nba' - Naive Bayes Adaptive

    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.

    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    bootstrap_sampling: bool, optional (default=True)
        If True, perform bootstrap sampling in the leaf nodes.

    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`. Only used when ``bootstrap_sampling=True`` to direct the bootstrap sampling.

    Notes
    -----
    The Hoeffding Adaptive Tree [1]_ uses ADWIN [2]_ to monitor performance of branches on the tree
    and to replace them with new branches when their accuracy decreases if the new branches are
    more accurate.

    The bootstrap sampling strategy is an improvement over the original Hoeffding Adaptive Tree
    algorithm. It is enabled by default since, in general, it results in better performance.

    References
    ----------
    .. [1] Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data streams."
       In International Symposium on Intelligent Data Analysis, pp. 249-260. Springer, Berlin,
       Heidelberg, 2009.
    .. [2] Bifet, Albert, and Ricard Gavaldà. "Learning from time-changing data with adaptive
       windowing." In Proceedings of the 2007 SIAM international conference on data mining,
       pp. 443-448. Society for Industrial and Applied Mathematics, 2007.

    Examples
    --------
    >>> from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
    >>> from skmultiflow.data import ConceptDriftStream
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>> # Setup the File Stream
    >>> stream = ConceptDriftStream(random_state=123456, position=25000)
    >>>
    >>> classifier = HoeffdingAdaptiveTreeClassifier()
    >>> evaluator = EvaluatePrequential(pretrain_size=200, max_samples=50000, batch_size=1,
    >>>                                 n_wait=200, max_time=1000, output_file=None,
    >>>                                 show_plot=True, metrics=['kappa', 'kappa_t', 'accuracy'])
    >>>
    >>> evaluator.evaluate(stream=stream, model=classifier)

    """
    # =============================================
    # == Hoeffding Adaptive Tree implementation ===
    # =============================================
    _ERROR_WIDTH_THRESHOLD = 300

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 bootstrap_sampling=True,
                 random_state=None):

        super().__init__(max_byte_size=max_byte_size,
                         memory_estimate_period=memory_estimate_period,
                         grace_period=grace_period,
                         split_criterion=split_criterion,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         binary_split=binary_split,
                         stop_mem_management=stop_mem_management,
                         remove_poor_atts=remove_poor_atts,
                         no_preprune=no_preprune,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes)
        self.alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self.bootstrap_sampling = bootstrap_sampling
        self.random_state = random_state
        self._tree_root = None

    def reset(self):
        self.alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self._tree_root = None

    # Override HoeffdingTreeClassifier
    def _partial_fit(self, X, y, sample_weight):
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        if isinstance(self._tree_root, InactiveLeaf):
            self._tree_root.learn_one(X, y, weight=sample_weight, tree=self)
        else:
            self._tree_root.learn_one(X, y, sample_weight, self, None, -1)

    def filter_instance_to_leaves(self, X, y, weight, split_parent, parent_branch,
                                  update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, y, weight, split_parent, parent_branch,
                                                  update_splitter_counts, nodes)
        return nodes

    # Override HoeffdingTreeClassifier
    def _get_votes_for_instance(self, X):
        result = {}
        if self._tree_root is not None:
            if isinstance(self._tree_root, InactiveLeaf):
                found_node = [self._tree_root.filter_instance_to_leaf(X, None, -1)]
            else:
                found_node = self.filter_instance_to_leaves(X, -np.inf, -np.inf, None, -1, False)
            for fn in found_node:
                if fn.parent_branch != -999:
                    leaf_node = fn.node
                    if leaf_node is None:
                        leaf_node = fn.parent
                    dist = leaf_node.predict_one(X, tree=self)
                    # add elements to dictionary
                    result = add_dict_values(result, dist, inplace=True)
        return result

    # Override HoeffdingTreeClassifier
    def _new_learning_node(self, initial_class_observations=None, is_active=True):
        if is_active:
            return AdaLearningNode(initial_class_observations, self.random_state)
        else:
            return InactiveLearningNodeMC(initial_class_observations)

    # Override HoeffdingTreeClassifier
    def _new_split_node(self, split_test, class_observations):
        return AdaSplitNode(split_test, class_observations, self.random_state)
