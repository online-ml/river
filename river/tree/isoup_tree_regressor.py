import typing
from copy import deepcopy

from river import base, tree

from .nodes.branch import HTBranch
from .nodes.isouptr_nodes import (
    LeafAdaptiveMultiTarget,
    LeafMeanMultiTarget,
    LeafModelMultiTarget,
)
from .split_criterion import IntraClusterVarianceReductionSplitCriterion
from .splitter import Splitter


class iSOUPTreeRegressor(tree.HoeffdingTreeRegressor, base.MultiOutputMixin):
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
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.EBSTSplitter` is used if `splitter` is `None`.
    min_samples_split
        The minimum number of samples every branch resulting from a split candidate must have
        to be considered valid.
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
        splitter: Splitter = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: int = 500,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
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
            splitter=splitter,
            min_samples_split=min_samples_split,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.split_criterion: str = "icvr"  # intra cluster variance reduction
        self.targets: set = set()

    @tree.HoeffdingTreeRegressor.split_criterion.setter
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
        return IntraClusterVarianceReductionSplitCriterion(
            min_samples_split=self.min_samples_split
        )

    def _new_leaf(self, initial_stats=None, parent=None):
        """Create a new learning node. The type of learning node depends on
        the tree configuration.
        """
        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        leaf_models = None
        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_models = {}
            else:
                try:
                    leaf_models = deepcopy(parent._leaf_models)  # noqa
                except AttributeError:
                    # Due to an emerging category in a nominal feature, a split node was reached
                    leaf_models = {}

        if self.leaf_prediction == self._TARGET_MEAN:
            return LeafMeanMultiTarget(initial_stats, depth, self.splitter)
        elif self.leaf_prediction == self._MODEL:
            return LeafModelMultiTarget(
                initial_stats, depth, self.splitter, leaf_models
            )
        else:  # adaptive learning node
            new_adaptive = LeafAdaptiveMultiTarget(
                initial_stats, depth, self.splitter, leaf_models
            )
            if parent is not None and isinstance(parent, LeafAdaptiveMultiTarget):
                new_adaptive._fmse_mean = parent._fmse_mean.copy()  # noqa
                new_adaptive._fmse_model = parent._fmse_model.copy()  # noqa

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

        super().learn_one(x, y, sample_weight=sample_weight)  # noqa

        return self

    def predict_one(
        self, x: dict
    ) -> typing.Dict[typing.Hashable, base.typing.RegTarget]:
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
        pred = {}
        if self._root is not None:
            if isinstance(self._root, HTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            pred = leaf.prediction(x, tree=self)
        return pred
