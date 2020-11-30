import typing
from copy import deepcopy

from river import base
from river.tree import HoeffdingTreeRegressor

from ._split_criterion import IntraClusterVarianceReductionSplitCriterion
from ._nodes import LearningNodeMeanMultiTarget
from ._nodes import LearningNodeModelMultiTarget
from ._nodes import LearningNodeAdaptiveMultiTarget


class iSOUPTreeRegressor(HoeffdingTreeRegressor, base.MultiOutputMixin):
    """Incremental Structured Output Prediction Tree (iSOUP-Tree) for multi-target regression.

    This is an implementation of the iSOUP-Tree proposed by A. Osojnik, P. Panov, and
    S. Džeroski [^1].

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to
        decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mean' - Target mean</br>
        - 'model' - Uses the model defined in `leaf_model`</br>
        - 'adaptive' - Chooses between 'mean' and 'model' dynamically</br>
    leaf_model
        The regression model(s) used to provide responses if `leaf_prediction='model'`. It can
        be either a regressor (in which case it is going to be replicated to all the targets)
        or a dictionary whose keys are target identifiers, and the values are instances of
        `river.base.Regressor.` If not provided, instances of `river.linear_model.LinearRegression`
        with the default hyperparameters are used for all the targets. If a dictionary is passed
        and not all target models are specified, copies from the first model match in the
        dictionary will be used to the remaining targets.
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
    Hoeffding trees rely on Attribute Observer (AO) algorithms to monitor input features
    and perform splits. Nominal features can be easily dealt with, since the partitions
    are well-defined. Numerical features, however, require more sophisticated solutions.
    Currently, only one AO is supported in `river` for regression trees:

    - The Extended Binary Search Tree (E-BST) uses an exhaustive algorithm to find split
    candidates, similarly to batch decision tree algorithms. It ends up storing all
    observations between split attempts. However, E-BST automatically removes bad split
    points periodically from its structure and, thus, alleviates the memory and time
    costs involved in its usage.

    References
    ----------
    [^1]: Aljaž Osojnik, Panče Panov, and Sašo Džeroski. "Tree-based methods for online
        multi-target regression." Journal of Intelligent Information Systems 50.2 (2018): 315-339.

    Examples
    --------
    >>> import numbers
    >>> from river import compose
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import tree

    >>> dataset = datasets.SolarFlare()

    >>> num = compose.SelectType(numbers.Number) | preprocessing.MinMaxScaler()
    >>> cat = compose.SelectType(str) | preprocessing.OneHotEncoder(sparse=False)

    >>> model = tree.iSOUPTreeRegressor(
    ...     grace_period=100,
    ...     leaf_prediction='model',
    ...     leaf_model={
    ...         'c-class-flares': linear_model.LinearRegression(l2=0.02),
    ...         'm-class-flares': linear_model.PARegressor(),
    ...         'x-class-flares': linear_model.LinearRegression(l2=0.1)
    ...     }
    ... )

    >>> pipeline = (num + cat) | model
    >>> metric = metrics.RegressionMultiOutput(metrics.MAE())

    >>> evaluate.progressive_val_score(dataset, pipeline, metric)
    MAE: 0.425929
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "model",
        leaf_model: typing.Union[base.Regressor, typing.Dict] = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list = None,
        attr_obs: str = "e-bst",
        attr_obs_params: dict = None,
        min_samples_split: int = 5,
        **kwargs
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            leaf_prediction=leaf_prediction,
            leaf_model=leaf_model,
            model_selector_decay=model_selector_decay,
            nominal_attributes=nominal_attributes,
            attr_obs=attr_obs,
            attr_obs_params=attr_obs_params,
            min_samples_split=min_samples_split,
            **kwargs
        )

        self.split_criterion: str = "icvr"  # intra cluster variance reduction
        self.targets: set = set()

    @HoeffdingTreeRegressor.leaf_prediction.setter
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

    @HoeffdingTreeRegressor.split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion == "vr":
            # Corner case due to parent class initialization
            split_criterion = "icvr"
        if split_criterion != "icvr":  # intra cluster variance reduction
            print(
                'Invalid split_criterion option "{}", will use default "{}"'.format(
                    split_criterion, "icvr"
                )
            )
            self._split_criterion = "icvr"
        else:
            self._split_criterion = split_criterion

    def _new_split_criterion(self):
        return IntraClusterVarianceReductionSplitCriterion(min_samples_split=self.min_samples_split)

    def _new_learning_node(self, initial_stats=None, parent=None):
        """Create a new learning node. The type of learning node depends on
        the tree configuration.
        """
        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_models = {}
            else:
                try:
                    leaf_models = deepcopy(parent._leaf_models)
                except AttributeError:
                    # Due to an emerging category in a nominal feature, a split node was reached
                    leaf_models = {}

        if self.leaf_prediction == self._TARGET_MEAN:
            return LearningNodeMeanMultiTarget(
                initial_stats, depth, self.attr_obs, self.attr_obs_params
            )
        elif self.leaf_prediction == self._MODEL:
            return LearningNodeModelMultiTarget(
                initial_stats, depth, self.attr_obs, self.attr_obs_params, leaf_models
            )
        else:  # adaptive learning node
            new_adaptive = LearningNodeAdaptiveMultiTarget(
                initial_stats, depth, self.attr_obs, self.attr_obs_params, leaf_models
            )
            if parent is not None:
                new_adaptive._fmse_mean = parent._fmse_mean.copy()
                new_adaptive._fmse_model = parent._fmse_model.copy()

            return new_adaptive

    def learn_one(
        self,
        x: dict,
        y: typing.Dict[typing.Hashable, base.typing.RegTarget],
        *,
        sample_weight: float = 1.0
    ) -> "iSOUPTreeRegressor":
        """Incrementally train the model with one sample.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Target values.
        sample_weight
            The weight of the passed sample.
        """
        # Update target set
        self.targets.update(y.keys())

        super().learn_one(x, y, sample_weight=sample_weight)

        return self

    def predict_one(self, x: dict) -> typing.Dict[typing.Hashable, base.typing.RegTarget]:
        """Predict the target values for a given instance.

        Parameters
        ----------
        x
            Sample for which we want to predict the labels.

        Returns
        -------
        dict
            Predicted target values.
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
                    return {
                        t: node.stats[t].mean.get() if t in node.stats else 0.0
                        for t in self.targets
                    }
            else:
                parent = found_node.parent
                return {
                    t: parent.stats[t].mean.get() if t in parent.stats else 0.0
                    for t in self.targets
                }
        else:
            # Model is empty
            return {}
