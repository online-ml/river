from copy import deepcopy

from river.tree import HoeffdingTreeRegressor
from river.utils.skmultiflow_utils import check_random_state

from river import base

from ._nodes import RandomLearningNodeMean
from ._nodes import RandomLearningNodeModel
from ._nodes import RandomLearningNodeAdaptive


class ARFHoeffdingTreeRegressor(HoeffdingTreeRegressor):
    """ARF Hoeffding Tree regressor.

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
        | Prediction mechanism used at leaves.
        | 'mean' - Target mean
        | 'model' - Uses the model defined in `leaf_model`
        | 'adaptive' - Chooses between 'mean' and 'model' dynamically
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
    max_features
        Number of randomly selected features to act as split candidates at each attempt.
    seed
            If int, seed is the seed used by the random number generator;
            If RandomState instance, seed is the random number generator;
            If None, the random number generator is the RandomState instance
            used by `np.random`.
    **kwargs
        Other parameters passed to `river.tree.BaseHoeffdingTree`.

    This is the base-estimator of the Adaptive Random Forest Regressor ensemble learner (see
    `river.ensemble.AdaptiveRandomForestRegressor`). This Hoeffding Tree Regressor includes a
    max_features parameter, which defines the number of randomly selected features to be
    considered at each split.
    """

    def __init__(self,
                 grace_period: int = 200,
                 max_depth: int = None,
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'model',
                 leaf_model: base.Regressor = None,
                 model_selector_decay: float = 0.95,
                 nominal_attributes: list = None,
                 max_features: int = 2,
                 seed=None,
                 **kwargs):
        super().__init__(grace_period=grace_period,
                         max_depth=max_depth,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         leaf_prediction=leaf_prediction,
                         leaf_model=leaf_model,
                         model_selector_decay=model_selector_decay,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self.max_features = max_features
        self.seed = seed
        self._rng = check_random_state(self.seed)

    def _new_learning_node(self, initial_stats=None, parent=None):
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """

        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        # Generate a random seed for the new learning node
        seed = self._rng.randint(0, 4294967295, dtype='u8')

        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_model = deepcopy(self.leaf_model)
            else:
                leaf_model = deepcopy(parent._leaf_model)

        if self.leaf_prediction == self._TARGET_MEAN:
            return RandomLearningNodeMean(initial_stats, depth, self.max_features, seed)
        elif self.leaf_prediction == self._MODEL:
            return RandomLearningNodeModel(
                initial_stats, depth, leaf_model, self.max_features, seed)
        else:  # adaptive learning node
            new_adaptive = RandomLearningNodeAdaptive(
                initial_stats, depth, leaf_model, self.max_features, seed)
            if parent is not None:
                new_adaptive._fmse_mean = parent._fmse_mean
                new_adaptive._fmse_model = parent._fmse_model

            return new_adaptive

    def new_instance(self):
        return self.__class__(**self._get_params())
