from operator import attrgetter
from copy import deepcopy

from river import base
from river import linear_model

from ._base_tree import BaseHoeffdingTree
from ._split_criterion import VarianceReductionSplitCriterion
from ._nodes import SplitNode
from ._nodes import LearningNode
from ._nodes import LearningNodeMean
from ._nodes import LearningNodeModel
from ._nodes import LearningNodeAdaptive


class HoeffdingTreeRegressor(BaseHoeffdingTree, base.Regressor):
    """Hoeffding Tree regressor.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mean' - Target mean</br>
        - 'model' - Uses the model defined in `leaf_model`</br>
        - 'adaptive' - Chooses between 'mean' and 'model' dynamically</br>
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
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    attr_obs
        The attribute observer (AO) used to monitor the target statistics of numeric
        features and perform splits. Parameters can be passed to the AOs (when supported)
        by using `attr_obs_params`. Valid options are:</br>
        - `'e-bst'`: Extended Binary Search Tree (E-BST). This AO has no parameters.</br>
        See notes for more information about the supported AOs.
    attr_obs_params
        Parameters passed to the numeric AOs. See `attr_obs` for more information.
    min_samples_split
        The minimum number of samples every branch resulting from a split candidate must have
        to be considered valid.
    kwargs
        Other parameters passed to `river.tree.BaseHoeffdingTree`.

    Notes
    -----
    The Hoeffding Tree Regressor (HTR) is an adaptation of the incremental tree algorithm of the
    same name for classification. Similarly to its classification counterpart, HTR uses the
    Hoeffding bound to control its split decisions. Differently from the classification algorithm,
    HTR relies on calculating the reduction of variance in the target space to decide among the
    split candidates. The smallest the variance at its leaf nodes, the more homogeneous the
    partitions are. At its leaf nodes, HTR fits either linear models or uses the target
    average as the predictor.

    Hoeffding trees rely on Attribute Observer (AO) algorithms to monitor input features
    and perform splits. Nominal features can be easily dealt with, since the partitions
    are well-defined. Numerical features, however, require more sophisticated solutions.
    Currently, only one AO is supported in `river` for regression trees:

    - The Extended Binary Search Tree (E-BST) uses an exhaustive algorithm to find split
    candidates, similarly to batch decision tree algorithms. It ends up storing all
    observations between split attempts. However, E-BST automatically removes bad split
    points periodically from its structure and, thus, alleviates the memory and time
    costs involved in its usage.

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
    ...     tree.HoeffdingTreeRegressor(
    ...         grace_period=100,
    ...         leaf_prediction='adaptive',
    ...         model_selector_decay=0.9
    ...     )
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.852902
    """

    _TARGET_MEAN = "mean"
    _MODEL = "model"
    _ADAPTIVE = "adaptive"
    _E_BST = "e-bst"
    _VALID_AO = [_E_BST]

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "model",
        leaf_model: base.Regressor = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list = None,
        attr_obs: str = "e-bst",
        attr_obs_params: dict = None,
        min_samples_split: int = 5,
        **kwargs,
    ):
        super().__init__(max_depth=max_depth, **kwargs)

        self._split_criterion: str = "vr"
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.leaf_prediction = leaf_prediction
        self.leaf_model = leaf_model if leaf_model else linear_model.LinearRegression()
        self.model_selector_decay = model_selector_decay
        self.nominal_attributes = nominal_attributes
        self.min_samples_split = min_samples_split

        if attr_obs not in self._VALID_AO:
            raise AttributeError(f'Invalid "attr_obs" option. Valid options are: {self._VALID_AO}')
        self.attr_obs = attr_obs
        self.attr_obs_params = attr_obs_params if attr_obs_params is not None else {}
        self.kwargs = kwargs

    @BaseHoeffdingTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in {self._TARGET_MEAN, self._MODEL, self._ADAPTIVE}:
            print(
                'Invalid leaf_prediction option "{}", will use default "{}"'.format(
                    leaf_prediction, self._MODEL
                )
            )
            self._leaf_prediction = self._MODEL
        else:
            self._leaf_prediction = leaf_prediction

    @BaseHoeffdingTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion != "vr":  # variance reduction
            print(
                "Invalid split_criterion option {}', will use default '{}'".format(
                    split_criterion, "vr"
                )
            )
            self._split_criterion = "vr"
        else:
            self._split_criterion = split_criterion

    def _new_split_criterion(self):
        return VarianceReductionSplitCriterion(min_samples_split=self.min_samples_split)

    def _new_learning_node(self, initial_stats=None, parent=None):
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """
        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_model = deepcopy(self.leaf_model)
            else:
                try:
                    leaf_model = deepcopy(parent._leaf_model)
                except AttributeError:
                    leaf_model = deepcopy(self.leaf_model)

        if self.leaf_prediction == self._TARGET_MEAN:
            return LearningNodeMean(initial_stats, depth, self.attr_obs, self.attr_obs_params)
        elif self.leaf_prediction == self._MODEL:
            return LearningNodeModel(
                initial_stats, depth, self.attr_obs, self.attr_obs_params, leaf_model
            )
        else:  # adaptive learning node
            new_adaptive = LearningNodeAdaptive(
                initial_stats, depth, self.attr_obs, self.attr_obs_params, leaf_model
            )
            if parent is not None:
                new_adaptive._fmse_mean = parent._fmse_mean
                new_adaptive._fmse_model = parent._fmse_model

            return new_adaptive

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Train the tree model on sample x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Target value for sample x.
        sample_weight
            The weight of the sample.

        Returns
        -------
        self
        """

        self._train_weight_seen_by_model += sample_weight

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._n_active_leaves = 1

        found_node = self._tree_root.filter_instance_to_leaf(x, None, -1)
        leaf_node = found_node.node

        if leaf_node is None:
            leaf_node = self._new_learning_node()
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

    def predict_one(self, x):
        """Predict the target value using one of the leaf prediction strategies.

        Parameters
        ----------
        x
            Instance for which we want to predict the target.

        Returns
        -------
        Predicted target value.

        """
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(x, None, -1)
            node = found_node.node
            if node is not None:
                if node.is_leaf():
                    return node.leaf_prediction(x, tree=self)
                else:
                    # The instance sorting ended up in a Split Node, since no branch was found
                    # for some of the instance's features. Use the mean prediction in this case
                    return node.stats.mean.get()
            else:
                parent = found_node.parent
                return parent.stats.mean.get()
        else:
            # Model is empty
            return 0.0

    def _attempt_to_split(self, node: LearningNode, parent: SplitNode, parent_idx: int):
        """Attempt to split a node.

        If the target's variance is high at the leaf node, then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the ratio between the merit of the second best split candidate and the merit of the
        best one is smaller than 1 minus the Hoeffding bound (or a tie breaking decision
        takes place), then:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        node
            The node to evaluate.
        parent
            The node's parent in the tree.
        parent_idx
            Parent node's branch index.

        """
        split_criterion = self._new_split_criterion()
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
            if best_suggestion.merit > 0.0 and (
                second_best_suggestion.merit / best_suggestion.merit < 1 - hoeffding_bound
                or hoeffding_bound < self.tie_threshold
            ):
                should_split = True
            if self.remove_poor_attrs:
                poor_attrs = set()
                best_ratio = second_best_suggestion.merit / best_suggestion.merit

                # Add any poor attribute to set
                for i in range(len(best_split_suggestions)):
                    if best_split_suggestions[i].split_test is not None:
                        split_attrs = best_split_suggestions[i].split_test.attrs_test_depends_on()
                        if len(split_attrs) == 1:
                            if (
                                best_split_suggestions[i].merit / best_suggestion.merit
                                < best_ratio - 2 * hoeffding_bound
                            ):
                                poor_attrs.add(split_attrs[0])
                for poor_att in poor_attrs:
                    node.disable_attribute(poor_att)
        if should_split:
            split_decision = best_split_suggestions[-1]
            if split_decision.split_test is None:
                # Preprune - null wins
                node.deactivate()
                self._n_inactive_leaves += 1
                self._n_active_leaves -= 1
            else:
                new_split = self._new_split_node(split_decision.split_test, node.stats, node.depth)
                for i in range(split_decision.num_splits()):
                    new_child = self._new_learning_node(
                        split_decision.resulting_stats_from_split(i), node
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
        elif (
            len(best_split_suggestions) >= 2
            and best_split_suggestions[-1].merit > 0
            and best_split_suggestions[-2].merit > 0
        ):
            last_check_ratio = best_split_suggestions[-2].merit / best_split_suggestions[-1].merit
            last_check_vr = best_split_suggestions[-1].merit

            node.manage_memory(split_criterion, last_check_ratio, last_check_vr, hoeffding_bound)
