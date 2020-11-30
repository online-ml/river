from operator import attrgetter

from river import base
from river.utils.skmultiflow_utils import normalize_values_in_dict

from ._base_tree import BaseHoeffdingTree
from ._split_criterion import GiniSplitCriterion
from ._split_criterion import InfoGainSplitCriterion
from ._split_criterion import HellingerDistanceCriterion
from ._nodes import SplitNode
from ._nodes import LearningNode
from ._nodes import LearningNodeMC
from ._nodes import LearningNodeNB
from ._nodes import LearningNodeNBA


class HoeffdingTreeClassifier(BaseHoeffdingTree, base.Classifier):
    """Hoeffding Tree or Very Fast Decision Tree classifier.

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
        List of Nominal attributes identifiers. If empty, then assume that all numeric
        attributes should be treated as continuous.
    attr_obs
        The Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Parameters can be passed to the AOs (when supported)
        by using `attr_obs_params`. Valid options are:</br>
        - `'bst'`: Binary Search Tree.</br>
        - `'gaussian'`: Gaussian observer. The `n_splits` used to query
         for split candidates can be adjusted (defaults to `10`).</br>
        - `'histogram'`: Histogram-based class frequency estimation.  The number of histogram
        bins (`n_bins` -- defaults to `256`) and the number of split point candidates to
        evaluate (`n_splits` -- defaults to `32`) can be adjusted.</br>
        See 'Notes' for more information about the AOs.
    attr_obs_params
        Parameters passed to the numeric AOs. See `attr_obs` for more information.
    kwargs
        Other parameters passed to `river.tree.BaseHoeffdingTree`.

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
    learner using infinitely many examples. Implementation based on MOA [^2].

    Hoeffding trees rely on Attribute Observer (AO) algorithms to monitor input features
    and perform splits. Nominal features can be easily dealt with, since the partitions
    are well-defined. Numerical features, however, require more sophisticated solutions.
    Currently, three AOs are supported in `river` for classification trees:

    - *Binary Search Tree (BST)*: uses an exhaustive algorithm to find split candidates,
    similarly to batch decision trees. It ends up storing all observations between split
    attempts. This AO is the most costly one in terms of memory and processing
    time; however, it tends to yield the most accurate results when using `leaf_prediction=mc`.
    It cannot be used to calculate the Probability Density Function (PDF) of the monitored
    feature due to its binary tree nature. Hence, leaf prediction strategies other than
    the majority class will end up effectively mimicing the majority class classifier.
    This AO has no parameters.</br>
    - *Gaussian Estimator*: Approximates the numeric feature distribution by using
    a Gaussian distribution per class. The Cumulative Distribution Function (CDF) necessary to
    calculate the entropy (and, consequently, the information gain), the gini index, and
    other split criteria is then calculated using the fit feature's distribution.</br>
    - *Histogram*: approximates the numeric feature distribution using an incrementally
    maintained histogram per class. It represents a compromise between the intensive
    resource usage of BST and the strong assumptions about the feature's distribution
    used in the Gaussian Estimator. Besides that, this AO sits in the middle between the
    previous two in terms of memory usage and running time. Note that the number of
    bins affects the probability density approximation required to use leaves with
    (adaptive) naive bayes models. Hence, Histogram tends to be less accurate than the
    Gaussian estimator when adaptive or naive bayes leaves are used.

    References
    ----------

    [^1]: G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    [^2]: Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------
    >>> from river import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.Agrawal(classification_function=0, seed=42)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.HoeffdingTreeClassifier(
    ...     grace_period=100,
    ...     split_confidence=1e-5,
    ...     nominal_attributes=['elevel', 'car', 'zipcode']
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 86.09%
    """

    _GINI_SPLIT = "gini"
    _INFO_GAIN_SPLIT = "info_gain"
    _HELLINGER = "hellinger"
    _MAJORITY_CLASS = "mc"
    _NAIVE_BAYES = "nb"
    _NAIVE_BAYES_ADAPTIVE = "nba"
    _BST = "bst"
    _GAUSSIAN = "gaussian"
    _HISTOGRAM = "histogram"
    _VALID_AO = [_BST, _GAUSSIAN, _HISTOGRAM]

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
        attr_obs: str = "gaussian",
        attr_obs_params: dict = None,
        **kwargs,
    ):

        super().__init__(max_depth=max_depth, **kwargs)
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        if attr_obs not in self._VALID_AO:
            raise AttributeError(f'Invalid "attr_obs" option. Valid options are: {self._VALID_AO}')
        self.attr_obs = attr_obs
        self.attr_obs_params = attr_obs_params if attr_obs_params is not None else {}
        self.kwargs = kwargs

        # To keep track of the observed classes
        self.classes: set = set()

    @BaseHoeffdingTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion not in [
            self._GINI_SPLIT,
            self._INFO_GAIN_SPLIT,
            self._HELLINGER,
        ]:
            print(
                "Invalid split_criterion option {}', will use default '{}'".format(
                    split_criterion, self._INFO_GAIN_SPLIT
                )
            )
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @BaseHoeffdingTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in [
            self._MAJORITY_CLASS,
            self._NAIVE_BAYES,
            self._NAIVE_BAYES_ADAPTIVE,
        ]:
            print(
                "Invalid leaf_prediction option {}', will use default '{}'".format(
                    leaf_prediction, self._NAIVE_BAYES_ADAPTIVE
                )
            )
            self._leaf_prediction = self._NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    def _new_learning_node(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return LearningNodeMC(initial_stats, depth, self.attr_obs, self.attr_obs_params)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return LearningNodeNB(initial_stats, depth, self.attr_obs, self.attr_obs_params)
        else:  # NAIVE BAYES ADAPTIVE (default)
            return LearningNodeNBA(initial_stats, depth, self.attr_obs, self.attr_obs_params)

    def _attempt_to_split(self, node: LearningNode, parent: SplitNode, parent_idx: int):
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
            best_split_suggestions = node.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter("merit"))
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(node.stats),
                    self.split_confidence,
                    node.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                    or hoeffding_bound < self.tie_threshold
                ):
                    should_split = True
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[
                                i
                            ].split_test.attrs_test_depends_on()
                            if len(split_atts) == 1:
                                if (
                                    best_suggestion.merit - best_split_suggestions[i].merit
                                    > hoeffding_bound
                                ):
                                    poor_atts.add(split_atts[0])
                    for poor_att in poor_atts:
                        node.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.split_test is None:
                    # Preprune - null wins
                    node.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    new_split = self._new_split_node(
                        split_decision.split_test, node.stats, node.depth
                    )

                    for i in range(split_decision.num_splits()):
                        new_child = self._new_learning_node(
                            split_decision.resulting_stats_from_split(i),
                            parent=new_split,
                        )
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

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Train the model on instance x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Class label for sample x.
        sample_weight
            Sample weight.

        Returns
        -------
        self

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

        if leaf_node.is_leaf():
            leaf_node.learn_one(x, y, sample_weight=sample_weight, tree=self)
            if self._growth_allowed and leaf_node.is_active():
                if leaf_node.depth >= self.max_depth:  # Max depth reached
                    leaf_node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = leaf_node.total_weight
                    weight_diff = weight_seen - leaf_node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        self._attempt_to_split(
                            leaf_node, found_node.parent, found_node.parent_branch
                        )
                        leaf_node.last_split_attempt_at = weight_seen
        else:
            current = leaf_node
            split_feat = current.split_test.attrs_test_depends_on()[0]
            # Split node encountered a previously unseen categorical value (in a multi-way test),
            # so there is no branch to sort the instance to
            if current.split_test.max_branches() == -1 and split_feat in x:
                # Creates a new branch to the new categorical value
                leaf_node = self._new_learning_node(parent=current)
                branch_id = current.split_test.add_new_branch(x[split_feat])
                current.set_child(branch_id, leaf_node)
                self._n_active_leaves += 1
                leaf_node.learn_one(x, y, sample_weight=sample_weight, tree=self)
            # The split feature is missing in the instance. Hence, we pass the new example
            # to the most traversed path in the current subtree
            else:
                nd = leaf_node
                # Keep traversing down the tree until a leaf is found
                while not nd.is_leaf():
                    path = max(
                        nd._children,
                        key=lambda c: nd._children[c].total_weight if nd._children[c] else 0.0,
                    )
                    most_traversed = nd.get_child(path)
                    # Pass instance to the most traversed path
                    if most_traversed is not None:
                        found_node = most_traversed.filter_instance_to_leaf(x, nd, path)
                        nd = found_node.node

                        if nd is None:
                            nd = self._new_learning_node(parent=found_node.parent)
                            found_node.parent.set_child(found_node.parent_branch, nd)
                            self._n_active_leaves += 1
                    else:
                        leaf = self._new_learning_node(parent=nd)
                        nd.set_child(path, leaf)
                        nd = leaf
                        self._n_active_leaves += 1

                # Learn from the sample
                nd.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in self.classes}
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(x, None, -1)
            node = found_node.node
            if node is None:
                node = found_node.parent

            if node.is_leaf():
                proba.update(node.leaf_prediction(x, tree=self))
            else:  # Corner case where a decision node is reached
                proba.update(normalize_values_in_dict(node.stats, inplace=False))
        return proba

    @property
    def _multiclass(self):
        return True
