from copy import deepcopy

from creme.tree import HoeffdingTreeRegressor
from creme.utils.skmultiflow_utils import check_random_state

from creme import base
from creme import linear_model

from ._nodes import RandomActiveLearningNodeMean
from ._nodes import InactiveLearningNodeMean
from ._nodes import RandomActiveLearningNodeModel
from ._nodes import InactiveLearningNodeModel
from ._nodes import RandomActiveLearningNodeAdaptive
from ._nodes import InactiveLearningNodeAdaptive


class ARFHoeffdingTreeRegressor(HoeffdingTreeRegressor):
    """ ARF Hoeffding Tree regressor.

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
        The regression model used to provide responses if `leaf_prediction='model'`.
    model_selector_decay
        The exponential decaying factor applied to the learning models' squared errors, that
        are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer
        to `1`, the more importance is going to be given to past observations. On the other hand,
        if its value approaches `0`, the recent observed errors are going to have more influence
        on the final decision.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    max_features
        Number of randomly selected features to act as split candidates at each attempt.
    random_state
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance
            used by `np.random`.
    **kwargs
        Other parameters passed to river.tree.DecisionTree.

    This is the base-estimator of the Adaptive Random Forest Regressor ensemble learner (see
    `river.ensemble.AdaptiveRandomForestRegressor`). This Hoeffding Tree Regressor includes a
    max_features parameter, which defines the number of randomly selected features to be
    considered at each split.
    """

    def __init__(self,
                 grace_period: int = 200,
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'model',
                 leaf_model: base.Regressor = linear_model.LinearRegression(),
                 model_selector_decay: float = 0.95,
                 nominal_attributes: list = None,
                 max_features: int = 2,
                 random_state=None,
                 **kwargs):
        super().__init__(grace_period=grace_period,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         leaf_prediction=leaf_prediction,
                         leaf_model=leaf_model,
                         model_selector_decay=model_selector_decay,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self.max_features = max_features
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)

    def _new_learning_node(self, initial_stats=None, parent=None, is_active=True):
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """
        if initial_stats is None:
            initial_stats = {}

        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        # Generate a random seed for the new learning node
        random_state = self._random_state.randint(0, 4294967295, dtype='u8')

        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                # TODO: change to appropriate 'clone' method
                leaf_model = self.leaf_model.__class__(**self.leaf_model._get_params())
            else:
                leaf_model = deepcopy(parent._leaf_model)

        if is_active:
            if self.leaf_prediction == self._TARGET_MEAN:
                return RandomActiveLearningNodeMean(
                    initial_stats, depth, self.max_features, random_state)
            elif self.leaf_prediction == self._MODEL:
                return RandomActiveLearningNodeModel(
                    initial_stats, depth, leaf_model, self.max_features, random_state)
            else:  # adaptive learning node
                new_adaptive = RandomActiveLearningNodeAdaptive(
                    initial_stats, depth, leaf_model, self.max_features, random_state)
                if parent is not None:
                    new_adaptive._fmse_mean = parent._fmse_mean
                    new_adaptive._fmse_model = parent._fmse_model

                return new_adaptive
        else:
            if self.leaf_prediction == self._TARGET_MEAN:
                return InactiveLearningNodeMean(initial_stats, depth)
            elif self.leaf_prediction == self._MODEL:
                return InactiveLearningNodeModel(initial_stats, depth, leaf_model)
            else:  # adaptive learning node
                new_adaptive = InactiveLearningNodeAdaptive(initial_stats, depth, leaf_model)
                if parent is not None:
                    new_adaptive._fmse_mean = parent._fmse_mean
                    new_adaptive._fmse_mean = parent._fmse_model

                return new_adaptive

    def reset(self):
        super().reset()
        self._random_state = check_random_state(self.random_state)

    def new_instance(self):
        return self.__class__(**self._get_params())
