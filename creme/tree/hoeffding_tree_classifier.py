from operator import attrgetter

from creme import base
from creme.utils.math import softmax

from ._base_tree import DecisionTree
from ._split_criterion import GiniSplitCriterion
from ._split_criterion import InfoGainSplitCriterion
from ._split_criterion import HellingerDistanceCriterion
from ._nodes import SplitNode
from ._nodes import LearningNode
from ._nodes import ActiveLeaf
from ._nodes import ActiveLearningNodeMC
from ._nodes import ActiveLearningNodeNB
from ._nodes import ActiveLearningNodeNBA
from ._nodes import InactiveLearningNodeMC


class HoeffdingTreeClassifier(DecisionTree, base.Classifier):
    """Hoeffding Tree or Very Fast Decision Tree classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    split_criterion
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
        | 'hellinger' - Helinger Distance
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    **kwargs
        Other parameters passed to river.tree.DecisionTree.

    Notes
    -----
    A Hoeffding Tree [^1] is an incremental, anytime decision tree induction algorithm that is
    capable of learning from massive data streams, assuming that the distribution generating
    examples does not change over time. Hoeffding trees exploit the fact that a small sample can
    often be enough to choose an optimal splitting attribute. This idea is supported mathematically
    by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of
    an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision
    tree learners is that it has sound guarantees of performance. Using the Hoeffding bound one
    can show that its output is asymptotically nearly identical to that of a non-incremental
    learner using infinitely many examples.

    Implementation based on MOA [^2].

    References:
    .. [1] G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    .. [2] Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------
    >>> from creme.stream import synth
    >>> from creme import evaluate
    >>> from creme import metrics
    >>> from creme import tree

    >>> dataset = datasets.Phishing()

    >>> model = tree.HoeffdingTreeClassifier(
    ...     grace_period=100,
    ...     split_confidence=1e-5,
    ...     split_criterion='gini'
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    """

    _GINI_SPLIT = 'gini'
    _INFO_GAIN_SPLIT = 'info_gain'
    _HELLINGER = 'hellinger'
    _MAJORITY_CLASS = 'mc'
    _NAIVE_BAYES = 'nb'
    _NAIVE_BAYES_ADAPTIVE = 'nba'

    def __init__(self,
                 grace_period: int = 200,
                 split_criterion: str = 'info_gain',
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'nba',
                 nb_threshold: int = 0,
                 nominal_attributes: list = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        self.classes: set = set()

    @DecisionTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion not in [self._GINI_SPLIT, self._INFO_GAIN_SPLIT, self._HELLINGER]:
            print("Invalid split_criterion option {}', will use default '{}'".
                  format(split_criterion, self._INFO_GAIN_SPLIT))
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @DecisionTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in [self._MAJORITY_CLASS, self._NAIVE_BAYES,
                                   self._NAIVE_BAYES_ADAPTIVE]:
            print("Invalid leaf_prediction option {}', will use default '{}'".
                  format(leaf_prediction, self._NAIVE_BAYES_ADAPTIVE))
            self._leaf_prediction = self._NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    def learn_one(self, x, y, *, sample_weight=1.):
        """Train the model on instance x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Class label for sample x.
        sample_weight
            Sample weight.

        Notes
        -----
        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.
        """

        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += sample_weight

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._n_active_leaves = 1

        found_node = self._tree_root.filter_instance_to_leaf(x, None, -1)
        leaf_node = found_node.node
        if leaf_node is None:
            leaf_node = self._new_learning_node(parent=found_node.parent)
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
            self._n_active_leaves += 1

        if isinstance(leaf_node, LearningNode):
            leaf_node.learn_one(x, y, sample_weight=sample_weight, tree=self)
            if self._growth_allowed and isinstance(leaf_node, ActiveLeaf):
                if leaf_node.depth >= self.max_depth:  # Max depth reached
                    self._deactivate_leaf(leaf_node, found_node.parent, found_node.parent_branch)
                else:
                    weight_seen = leaf_node.total_weight
                    weight_diff = weight_seen - leaf_node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        self._attempt_to_split(leaf_node, found_node.parent,
                                               found_node.parent_branch)
                        leaf_node.last_split_attempt_at = weight_seen
        # Split node encountered a previously unseen categorical value (in a multi-way test),
        # so there is no branch to sort the instance to
        elif isinstance(leaf_node, SplitNode):
            # Creates a new branch to the new categorical value
            current = leaf_node
            leaf_node = self._new_learning_node(parent=current)
            branch_id = current.split_test.add_new_branch(
                x[current.split_test.get_atts_test_depends_on()[0]])
            current.set_child(branch_id, leaf_node)
            self._n_active_leaves += 1
            leaf_node.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    def predict_proba_one(self, x):
        """Predict probabilities of all label of the x instance.

        Parameters
        ----------
        x
            Instance for which we want to predict the label.
        """

        proba = {c: 0. for c in self.classes}
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(x, None, -1)
            leaf = found_node.node
            if leaf is None:
                leaf = found_node.parent

            pred = leaf.predict_one(x, tree=self) if not isinstance(leaf, SplitNode) \
                else leaf.stats
            proba.update(pred)
            proba = softmax(proba)
        return proba

    def _new_learning_node(self, initial_stats=None, parent=None, is_active=True):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if is_active:
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return ActiveLearningNodeMC(initial_stats, depth)
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return ActiveLearningNodeNB(initial_stats, depth)
            else:  # NAIVE BAYES ADAPTIVE (default)
                return ActiveLearningNodeNBA(initial_stats, depth)
        else:
            return InactiveLearningNodeMC(initial_stats, depth)

    def _attempt_to_split(self, node: ActiveLeaf, parent: SplitNode, parent_idx: int):
        """Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        node
            The node to evaluate.
        parent
            The node's parent.
        parent_idx
            Parent node's branch index.
        """
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == self._GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == self._INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            elif self._split_criterion == self._HELLINGER:
                split_criterion = HellingerDistanceCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()
            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter('merit'))
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(split_criterion.get_range_of_merit(
                    node.stats), self.split_confidence, node.total_weight)
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):
                    should_split = True
                if self.remove_poor_atts:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.\
                                get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if (best_suggestion.merit - best_split_suggestions[i].merit
                                        > hoeffding_bound):
                                    poor_atts.add(split_atts[0])
                    for poor_att in poor_atts:
                        node.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.split_test is None:
                    # Preprune - null wins
                    self._deactivate_leaf(node, parent, parent_idx)
                else:
                    new_split = self._new_split_node(split_decision.split_test, node.stats,
                                                     node.depth)

                    for i in range(split_decision.num_splits()):
                        new_child = self._new_learning_node(
                            split_decision.resulting_stats_from_split(i), parent=new_split)
                        new_split.set_child(i, new_child)
                    self._n_active_leaves -= 1
                    self._n_decision_nodes += 1
                    self._n_active_leaves += split_decision.num_splits()
                    if parent is None:
                        self._tree_root = new_split
                    else:
                        parent.set_child(parent_idx, new_split)

                # Manage memory
                self._enforce_size_limit()
