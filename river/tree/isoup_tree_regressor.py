import typing
from copy import deepcopy

from river import base
from river import linear_model
from river.tree import HoeffdingTreeRegressor

from ._split_criterion import IntraClusterVarianceReductionSplitCriterion
from ._nodes import ActiveLearningNodeMean
from ._nodes import ActiveLearningNodeModelMultiTarget
from ._nodes import ActiveLearningNodeAdaptiveMultiTarget
from ._nodes import InactiveLearningNodeMean
from ._nodes import InactiveLearningNodeModelMultiTarget
from ._nodes import InactiveLearningNodeAdaptiveMultiTarget


class iSOUPTreeRegressor(HoeffdingTreeRegressor, base.MultiOutputMixin):
    """Incremental Structured Output Prediction Tree (iSOUP-Tree) for multi-target regression.

    This is an implementation of the iSOUP-Tree proposed by A. Osojnik, P. Panov, and
    S. Džeroski [^1].

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to
        decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        | Prediction mechanism used at leafs.
        | 'mean' - Target mean
        | 'model' - Uses the model defined in `leaf_model`
        | 'adaptive' - Chooses between 'mean' and 'model' dynamically
    leaf_model
        The regression model(s) used to provide responses if `leaf_prediction='model'`. It can
        be either a regressor (in which case it is going to be replicated to all the targets)
        or a dictionary whose keys are target identifiers, and the values are instances of
        `river.base.Regressor.` If not provided instances of `river.linear_model.LinearRegression`
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
    **kwargs
        Other parameters passed to river.tree.DecisionTree.

    References
    ----------
    .. [1] Aljaž Osojnik, Panče Panov, and Sašo Džeroski. "Tree-based methods for online
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

    def __init__(self,
                 grace_period: int = 200,
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'model',
                 leaf_model: typing.Union[base.Regressor, dict] = None,
                 model_selector_decay: float = 0.95,
                 nominal_attributes: list = None,
                 **kwargs):
        super().__init__(grace_period=grace_period,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         leaf_prediction=leaf_prediction,
                         leaf_model=leaf_model,
                         model_selector_decay=model_selector_decay,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self.split_criterion: str = 'icvr'   # intra cluster variance reduction
        self._targets: set = set()

    @HoeffdingTreeRegressor.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in {self._TARGET_MEAN, self._MODEL, self._ADAPTIVE}:
            print('Invalid leaf_prediction option "{}", will use default "{}"'.format(
                leaf_prediction, self._MODEL))
            self._leaf_prediction = self._MODEL
        else:
            self._leaf_prediction = leaf_prediction

    @HoeffdingTreeRegressor.split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion == 'vr':
            # Corner case due to parent class initialization
            split_criterion = 'icvr'
        if split_criterion != 'icvr':   # intra cluster variance reduction
            print('Invalid split_criterion option "{}", will use default "{}"'
                  .format(split_criterion, 'icvr'))
            self._split_criterion = 'icvr'
        else:
            self._split_criterion = split_criterion

    def _new_split_criterion(self):
        return IntraClusterVarianceReductionSplitCriterion()

    def _new_learning_node(self, initial_stats=None, parent=None, is_active=True):
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

        if is_active:
            if self.leaf_prediction == self._TARGET_MEAN:
                return ActiveLearningNodeMean(initial_stats, depth)
            elif self.leaf_prediction == self._MODEL:
                return ActiveLearningNodeModelMultiTarget(initial_stats, depth, leaf_models)
            else:  # adaptive learning node
                new_adaptive = ActiveLearningNodeAdaptiveMultiTarget(initial_stats, depth,
                                                                     leaf_models)
                if parent is not None:
                    new_adaptive._fmse_mean = parent._fmse_mean.copy()
                    new_adaptive._fmse_model = parent._fmse_model.copy()

                return new_adaptive
        else:
            if self.leaf_prediction == self._TARGET_MEAN:
                return InactiveLearningNodeMean(initial_stats, depth)
            elif self.leaf_prediction == self._MODEL:
                return InactiveLearningNodeModelMultiTarget(initial_stats, depth, leaf_models)
            else:  # adaptive learning node
                new_adaptive = InactiveLearningNodeAdaptiveMultiTarget(initial_stats, depth,
                                                                       leaf_models)
                if parent is not None:
                    new_adaptive._fmse_mean = parent._fmse_mean.copy()
                    new_adaptive._fmse_mean = parent._fmse_model.copy()

                return new_adaptive

    def learn_one(self, x: dict, y: typing.Dict[typing.Union[str, int], base.typing.RegTarget], *,
                  sample_weight: float = 1.) -> 'iSOUPTreeRegressor':
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
        self._targets.update(y.keys())

        super().learn_one(x, y, sample_weight=sample_weight)

        return self

    def predict_one(self, x: dict) -> typing.Union[typing.Dict[typing.Union[str, int], base.typing.RegTarget], None]:
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
                    return node.predict_one(x, tree=self)
                else:
                    # The instance sorting ended up in a Split Node, since no branch was found
                    # for some of the instance's features. Use the mean prediction in this case
                    return {
                        t: node.stats[t].mean.get() if t in node.stats else 0.
                        for t in self._targets
                    }
            else:
                parent = found_node.parent
                return {
                    t: parent.stats[t].mean.get() if t in parent.stats else 0.
                    for t in self._targets
                }
        else:
            # Model is empty
            return {}
