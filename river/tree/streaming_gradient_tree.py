import abc
import collections
import functools
import math
import numbers
import typing

from scipy.stats import f as FTest

from river import base
from river import stats

from ._nodes import SGTNode
from ._objective import BaseObjective, BinaryCrossEntropyObjective, SquaredErrorObjective
from ._utils import GradHess


class BaseStreamingGradientTree(base.Estimator, metaclass=abc.ABCMeta):
    """ Streaming Gradient Tree for classification.

    This implementation enhances the original proposal [^1] by using an incremental strategy to
    discretize numerical features dynamically, rather than relying on a calibration set and
    parameterized number of bins. The strategy used is an adaptation of the Quantile Observer
    (QO) [^2]. Different bin size setting policies are available for selection.
    They directly related to number of split candidates the tree is going to explore, and thus,
    how accurate its split decisions are going to be. Besides, the number of stored bins per
    feature is directly related to the tree's memory usage and runtime.

    Parameters
    ----------
    delta
        Define the significance level of the F-tests performed to decide upon creating splits
        or updating predictions.
    grace_period
        Interval between split attempts or prediction updates.
    init_pred
        Initial value predicted by the tree.
    max_depth
        The maximum depth the tree might reach. If set to `None`, the trees will grow
        indefinitely.
    lambda_value
        Positive float value used to impose a penalty over the tree's predictions and force
        them become smaller. The greater the lambda value, the more constrained are the
        predictions.
    gamma
        Positive float value used to impose a penalty over the tree's splits and force them to
        be avoided when possible. The greater the gamma value, the smaller the chance of a
        split occurring.
    nominal_attributes
        List with identifiers of the nominal attributes. If None, all features containing
        numbers are assumed to be numeric.
    quantization_strategy
        Defines the policy used to define the quantization radius applied on numerical
        attributes' discretization. Each time a new leaf is created, a radius, or
        discretization interval, is assigned to each feature in order to quantize it.
        The smaller the interval used, the more points are going to be stored by the tree
        between splits. Although using more points ought to increase the memory footprint and
        runtime, the split candidates potentially are going to be better. When the trees are
        created and no data was observed, the initial radius is defined by the parameter
        `default_radius`. The next nodes are going to use the policy define by this
        parameter.</br>
        Valid values are:
        - 'stddiv': use the standard deviation of the feature divided by `quantization_radius_div`
        as the radius.</br>
        - 'constant': use the value defined in 'default_radius' as the static quantization
        interval.
    quantization_radius_div
        Value by which the standard deviation of numeric features is going to be divided to define
        the quantization radius. Only used when `quantization_strategy='stddiv'`.
    default_radius
        Quantization radius used when the tree is created and no data is observed or when
        `quantization_strategy='constant'`.

    References
    ---------
    [^1]: Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees.
    In Asian Conference on Machine Learning (pp. 1094-1109).
    [^2]: Mastelini, S.M. and de Carvalho, A.C.P.D.L.F., 2020. Using dynamical quantization to
    perform split attempts in online tree regressors. arXiv preprint arXiv:2012.00083.
    """

    _STD_DIV = 'stddiv'
    _CONSTANT_RAD = 'constant'

    _VALID_QUANTIZATION_POLICIES = [
        _STD_DIV,
        _CONSTANT_RAD
    ]

    def __init__(self, delta, grace_period, init_pred, max_depth, lambda_value, gamma,
                 nominal_attributes, quantization_strategy, quantization_radius_div,
                 default_radius):

        self.delta = delta
        self.grace_period = grace_period
        self.init_pred = init_pred
        self.max_depth = max_depth if max_depth else math.inf

        if lambda_value < 0.0:
            raise ValueError('Invalid value: "lambda_value" must be positive.')

        if gamma < 0.0:
            raise ValueError('Invalid value: "gamma" must be positive.')

        self.lambda_value = lambda_value
        self.gamma = gamma
        self.nominal_attributes = set(nominal_attributes) if nominal_attributes else set()

        if quantization_strategy not in self._VALID_QUANTIZATION_POLICIES:
            raise ValueError('Invalid "quantization_strategy": {}'.format(quantization_strategy))
        self.quantization_strategy = quantization_strategy
        self.quantization_radius_div = quantization_radius_div
        self.default_radius = default_radius

        self._root = SGTNode(prediction=self.init_pred)

        # set used to check whether categorical feature has been already split
        self._split_features = set()
        self._features_mean_var = collections.defaultdict(functools.partial(stats.Var))
        self._n_splits = 0
        self._n_node_updates = 0
        self._n_observations = 0
        self._depth = 0

        self._objective: BaseObjective

    def _target_transform(self, y):
        """Apply transformation to the raw target input.

        Different strategies are used for classification and regression. By default, implement
        an identity function.
        """
        return y

    def _update_features_stats(self, x: dict):
        """ Update inner tree statistics with a new observation.

        Keep estimators of the features' mean and variance.

        Parameters
        ----------
        x
            The incoming instance.
        """

        for feat_id, x_val in x.items():
            # Skip nominal attributes
            if not isinstance(x_val, numbers.Number) or feat_id in self.nominal_attributes:
                self.nominal_attributes.add(feat_id)
                continue
            self._features_mean_var[feat_id].update(x_val)

    def _get_quantization_radius(self, feat_id: typing.Hashable):
        """ Get the quantization radius for a given input feature. """
        if self.quantization_strategy == self._CONSTANT_RAD:
            return self.default_radius

        if self._n_observations < 2:
            # Default radius for quantization: might create too many bins at first
            return self.default_radius

        std_ = math.sqrt(self._features_mean_var[feat_id].get())

        if math.isclose(std_, 0):
            return self.default_radius

        return std_ / self.quantization_radius_div

    def _update_tree(self, x: dict, grad_hess: GradHess, w: float):
        """ Update Streaming Gradient Tree with a single instance. """

        leaf = self._root.sort_instance(x)
        leaf.update(x, grad_hess, self, w)

        if (leaf.total_weight - leaf.last_split_attempt_at < self.grace_period
                or leaf.depth >= self.max_depth):
            return

        # Update split attempt data
        leaf.last_split_attempt_at = leaf.total_weight
        best_split = leaf.find_best_split(self)

        p = self._compute_p_value(best_split, leaf.total_weight)

        if p < self.delta and best_split.loss_mean < 0:
            leaf.apply_split(best_split, self)

    @staticmethod
    def _compute_p_value(split, n_observations):
        # Null hypothesis: expected loss is zero
        # Alternative hypothesis: expected loss is not zero

        F = n_observations * (split.loss_mean * split.loss_mean) / split.loss_var \
            if split.loss_var > 0.0 else None

        if F is None:
            return 1.0

        return 1 - FTest.cdf(F, 1, n_observations - 1)

    def learn_one(self, x, y, *, w=1.0):
        self._update_features_stats(x)

        raw_pred = self._raw_prediction(x)
        label = self._target_transform(y)

        grad_hess = self._objective.compute_derivatives(label, raw_pred)   # noqa

        # Update the tree with the gradient/hessian info
        self._update_tree(x, grad_hess, w)

    def _raw_prediction(self, x):
        """ Obtain a raw prediction for a single instance. """

        pred = self._root.sort_instance(x).leaf_prediction()
        return self._objective.transfer(pred)    # noqa

    @property
    def n_nodes(self):
        n = 0
        to_visit = [self._root]
        while len(to_visit) > 0:
            node = to_visit.pop(0)
            n += 1

            if node.children is not None:
                for child in node.children.values():
                    to_visit.append(child)
        return n

    @property
    def n_splits(self):
        return self._n_splits

    @property
    def n_node_updates(self):
        return self._n_node_updates

    @property
    def n_observations(self):
        return self._n_observations

    @property
    def depth(self):
        return self._depth


class StreamingGradientTreeClassifier(BaseStreamingGradientTree, base.Classifier):
    def __init__(self, delta: float = 1e-7, grace_period: int = 200, init_pred: float = 0.0,
                 max_depth: typing.Optional[int] = None, lambda_value: float = 0.1,
                 gamma: float = 1.0, nominal_attributes: typing.Optional[typing.List] = None,
                 quantization_strategy: str = 'stddiv', quantization_radius_div: float = 2.0,
                 default_radius: float = 0.01):

        super().__init__(
            delta=delta,
            grace_period=grace_period,
            init_pred=init_pred,
            max_depth=max_depth,
            lambda_value=lambda_value,
            gamma=gamma,
            nominal_attributes=nominal_attributes,
            quantization_strategy=quantization_strategy,
            quantization_radius_div=quantization_radius_div,
            default_radius=default_radius
        )

        self._objective = BinaryCrossEntropyObjective()

    def _target_transform(self, y):
        return float(y)

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        t_proba = self._objective.transfer(self._root.sort_instance(x).leaf_prediction())

        return {True: t_proba, False: 1 - t_proba}


class StreamingGradientTreeRegressor(BaseStreamingGradientTree, base.Regressor):
    def __init__(self, delta: float = 1e-7, grace_period: int = 200, init_pred: float = 0.0,
                 max_depth: typing.Optional[int] = None, lambda_value: float = 0.1,
                 gamma: float = 1.0, nominal_attributes: typing.Optional[typing.List] = None,
                 quantization_strategy: str = 'stddiv', quantization_radius_div: float = 2.0,
                 default_radius: float = 0.01):

        super().__init__(
            delta=delta,
            grace_period=grace_period,
            init_pred=init_pred,
            max_depth=max_depth,
            lambda_value=lambda_value,
            gamma=gamma,
            nominal_attributes=nominal_attributes,
            quantization_strategy=quantization_strategy,
            quantization_radius_div=quantization_radius_div,
            default_radius=default_radius
        )

        self._objective = SquaredErrorObjective()

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        return self._root.sort_instance(x).leaf_prediction()
