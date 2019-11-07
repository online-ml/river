import copy

from skmultiflow.trees import HoeffdingTree
from skmultiflow.trees.nodes import InactiveLearningNode
from skmultiflow.trees.nodes import AdaLearningNode
from skmultiflow.trees.nodes import AdaSplitNode

import numpy as np


MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'
ERROR_WIDTH_THRESHOLD = 300


class HAT(HoeffdingTree):
    """ Hoeffding Adaptive Tree.

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

    Notes
    -----
    The Hoeffding Adaptive Tree [1]_ uses ADWIN [2]_ to monitor performance of branches on the tree and to replace them
    with new branches when their accuracy decreases if the new branches are more accurate.

    References
    ----------
    .. [1] Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data streams."\
       In International Symposium on Intelligent Data Analysis, pp. 249-260. Springer, Berlin, Heidelberg, 2009.
    .. [2] Bifet, Albert, and Ricard Gavaldà. "Learning from time-changing data with adaptive windowing."\
       In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448. Society for Industrial\
       and Applied Mathematics, 2007.

    Examples
    --------
    >>> from skmultiflow.trees.hoeffding_adaptive_tree import HAT
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> # Setup the File Stream
    >>> stream = FileStream("/skmultiflow/data/datasets/covtype.csv", -1, 1)
    >>> stream.prepare_for_use()
    >>>
    >>> classifier = HAT()
    >>> evaluator = EvaluatePrequential(pretrain_size=200, max_samples=50000, batch_size=1, n_wait=200, max_time=1000,
    >>>                                 output_file=None, show_plot=True, metrics=['kappa', 'kappa_t', 'performance'])
    >>>
    >>> evaluator.evaluate(stream=stream, model=classifier)

    """
    # =============================================
    # == Hoeffding Adaptive Tree implementation ===
    # =============================================

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
                 nominal_attributes=None):

        super(HAT, self).__init__(max_byte_size=max_byte_size,
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
        self._tree_root = None

    def reset(self):
        self.alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self._tree_root = None

    # Override HoeffdingTree
    def _partial_fit(self, X, y, sample_weight):
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        if isinstance(self._tree_root, InactiveLearningNode):
            self._tree_root.learn_from_instance(X, y, sample_weight, self)
        else:
            self._tree_root.learn_from_instance(X, y, sample_weight, self, None, -1)

    def filter_instance_to_leaves(self, X, y, weight, split_parent, parent_branch, update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, y, weight, split_parent, parent_branch,
                                                  update_splitter_counts, nodes)
        return nodes

    # Override HoeffdingTree
    def get_votes_for_instance(self, X):
        result = {}
        if self._tree_root is not None:
            if isinstance(self._tree_root, InactiveLearningNode):
                found_node = [self._tree_root.filter_instance_to_leaf(X, None, -1)]
            else:
                found_node = self.filter_instance_to_leaves(X, -np.inf, -np.inf, None, -1, False)
            for fn in found_node:
                if fn.parent_branch != -999:
                    leaf_node = fn.node
                    if leaf_node is None:
                        leaf_node = fn.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.update(dist)  # add elements to dictionary
        return result

    # Override HoeffdingTree
    def _new_learning_node(self, initial_class_observations=None):
        return AdaLearningNode(initial_class_observations)

    # Override HoeffdingTree
    def new_split_node(self, split_test, class_observations):
        return AdaSplitNode(split_test, class_observations)
