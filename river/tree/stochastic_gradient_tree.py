import abc
import collections
import functools
import math
import numbers
import typing

from scipy.stats import f as f_dist

from river import base, stats

from ._nodes import SGTLearningNode, SGTSplit
from ._objective import (
    BaseObjective,
    BinaryCrossEntropyObjective,
    SquaredErrorObjective,
)
from ._utils import GradHess


class BaseStochasticGradientTree(base.Estimator, metaclass=abc.ABCMeta):
    """ Base Stochastic Gradient Tree (SGT) class.

    This class defines the main characteristics that are shared by the different SGT
    implementations.

    """

    _STD_DIV = "stddiv"
    _CONSTANT_RAD = "constant"

    _VALID_QUANTIZATION_POLICIES = [_STD_DIV, _CONSTANT_RAD]

    def __init__(
        self,
        delta,
        grace_period,
        init_pred,
        max_depth,
        lambda_value,
        gamma,
        nominal_attributes,
        quantization_strategy,
    ):

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
        self.nominal_attributes = (
            set(nominal_attributes) if nominal_attributes else set()
        )

        if 2 <= len(quantization_strategy) <= 3:
            if len(quantization_strategy) == 2:
                self.quantization_strategy = (*quantization_strategy, 0.01)
            else:
                self.quantization_strategy = quantization_strategy
        else:
            raise ValueError(
                'Invalid number of parameters passed to "quantization_strategy".'
                "Two or three values must be passed."
            )

        if self.quantization_strategy[0] not in self._VALID_QUANTIZATION_POLICIES:
            raise ValueError(
                f'Invalid "quantization_strategy": {self.quantization_strategy[0]}'
            )

        self._root = SGTLearningNode(prediction=self.init_pred)

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
            if (
                not isinstance(x_val, numbers.Number)
                or feat_id in self.nominal_attributes
            ):
                self.nominal_attributes.add(feat_id)
                continue
            self._features_mean_var[feat_id].update(x_val)

    def _get_quantization_radius(self, feat_id: typing.Hashable):
        """ Get the quantization radius for a given input feature. """
        if self.quantization_strategy[0] == self._CONSTANT_RAD:
            return self.quantization_strategy[1]

        if self._n_observations < 2:
            # Default radius for quantization: might create too many bins at first
            return self.quantization_strategy[2]

        std_ = math.sqrt(self._features_mean_var[feat_id].get())

        if math.isclose(std_, 0):
            return self.quantization_strategy[2]

        return std_ / self.quantization_strategy[1]

    def _update_tree(self, x: dict, grad_hess: GradHess, w: float):
        """ Update Stochastic Gradient Tree with a single instance. """

        leaf = self._root.sort_instance_to_leaf(x)
        leaf.update(x, grad_hess, self, w)

        if leaf.total_weight - leaf.last_split_attempt_at < self.grace_period:
            return

        # Update split attempt data
        leaf.last_split_attempt_at = leaf.total_weight

        # If the maximum depth is reached, attempt to apply a "null split", i.e., update the
        # prediction value
        if leaf.depth >= self.max_depth:
            # Null split: update the prediction using the new gradient information
            best_split = SGTSplit()
            best_split.delta_pred = leaf.delta_prediction(
                leaf.update_stats.mean, self.lambda_value
            )
            dlms = leaf.update_stats.delta_loss_mean_var(best_split.delta_pred)
            best_split.loss_mean = dlms.mean.get()
            best_split.loss_var = dlms.get()
        else:  # Proceed with the standard split attempt procedure
            best_split = leaf.find_best_split(self)

        p = self._compute_p_value(best_split, leaf.total_weight)

        if p < self.delta and best_split.loss_mean < 0:
            leaf.apply_split(best_split, self)

    @staticmethod
    def _compute_p_value(split, n_observations):
        # Null hypothesis: expected loss is zero
        # Alternative hypothesis: expected loss is not zero

        f_value = (
            n_observations * (split.loss_mean * split.loss_mean) / split.loss_var
            if split.loss_var > 0.0
            else None
        )

        if f_value is None:
            return 1.0

        return 1 - f_dist.cdf(f_value, 1, n_observations - 1)

    def learn_one(self, x, y, *, w=1.0):
        self._update_features_stats(x)

        y_pred_raw = self._raw_prediction(x)
        y_true_trs = self._target_transform(y)

        grad_hess = self._objective.compute_derivatives(y_true_trs, y_pred_raw)  # noqa

        # Update the tree with the gradient/hessian info
        self._update_tree(x, grad_hess, w)

        return self

    def _raw_prediction(self, x):
        """ Obtain a raw prediction for a single instance. """

        y_pred_raw = self._root.sort_instance_to_leaf(x).leaf_prediction()
        return self._objective.transfer(y_pred_raw)  # noqa

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


class StochasticGradientTreeClassifier(BaseStochasticGradientTree, base.Classifier):
    """Stochastic Gradient Tree for classification.

    Binary decision tree classifier that minimizes the binary cross-entropy to guide its growth.

    Stochastic Gradient Trees (SGT) directly minimize a loss function to guide tree growth and
    update their predictions. Thus, they differ from other incrementally tree learners that do
    not directly optimize the loss, but a data impurity-related heuristic.

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
        them to become smaller. The greater the lambda value, the more constrained are the
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
        created and no data was observed, an initial radius must be applied to quantize the
        incoming features. The feature quantization is controlled by a tuple containing two or
        three positions, in the following order:</br>
        1. Quantization policy: can be either `'stddiv'` or `'constant'`. The former uses a
        proportion of the features' standard deviation as radius, while the latter uses a constant
        value.</br>
        2. The standard deviation proportion (value by which the standard deviation will be
        divided) or a constant radius value, as defined in the first position of the tuple.</br>
        3. [Optional] If the quantization policy is `'stddiv'`, an inicial radius value must be
        passed to be used as a cold-start for the feature quantization. If not provided, `0.01`
        is assumed as default.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Phishing()
    >>> model = tree.StochasticGradientTreeClassifier()
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 82.32%

    Notes
    -----
    This implementation enhances the original proposal [^1] by using an incremental strategy to
    discretize numerical features dynamically, rather than relying on a calibration set and
    parameterized number of bins. The strategy used is an adaptation of the Quantile Observer
    (QO) [^2]. Different bin size setting policies are available for selection.
    They directly related to number of split candidates the tree is going to explore, and thus,
    how accurate its split decisions are going to be. Besides, the number of stored bins per
    feature is directly related to the tree's memory usage and runtime.

    References
    ---------
    [^1]: Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees.
    In Asian Conference on Machine Learning (pp. 1094-1109).
    [^2]: Mastelini, S.M. and de Carvalho, A.C.P.D.L.F., 2020. Using dynamical quantization to
    perform split attempts in online tree regressors. arXiv preprint arXiv:2012.00083.

    """

    def __init__(
        self,
        delta: float = 1e-7,
        grace_period: int = 200,
        init_pred: float = 0.0,
        max_depth: typing.Optional[int] = None,
        lambda_value: float = 0.1,
        gamma: float = 1.0,
        nominal_attributes: typing.Optional[typing.List] = None,
        quantization_strategy: typing.Tuple[str, float, typing.Optional[float]] = (
            "stddiv",
            3.0,
        ),
    ):

        super().__init__(
            delta=delta,
            grace_period=grace_period,
            init_pred=init_pred,
            max_depth=max_depth,
            lambda_value=lambda_value,
            gamma=gamma,
            nominal_attributes=nominal_attributes,
            quantization_strategy=quantization_strategy,
        )

        self._objective = BinaryCrossEntropyObjective()

    def _target_transform(self, y):
        return float(y)

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        t_proba = self._objective.transfer(
            self._root.sort_instance_to_leaf(x).leaf_prediction()
        )

        return {True: t_proba, False: 1 - t_proba}


class StochasticGradientTreeRegressor(BaseStochasticGradientTree, base.Regressor):
    """Stochastic Gradient Tree for regression.

    Incremental decision tree regressor that minimizes the mean square error to guide its growth.

    Stochastic Gradient Trees (SGT) directly minimize a loss function to guide tree growth and
    update their predictions. Thus, they differ from other incrementally tree learners that do
    not directly optimize the loss, but a data impurity-related heuristic.

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
        them to become smaller. The greater the lambda value, the more constrained are the
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
        created and no data was observed, an initial radius must be applied to quantize the
        incoming features. The feature quantization is controlled by a tuple containing two or
        three positions, in the following order:</br>
        1. Quantization policy: can be either `'stddiv'` or `'constant'`. The former uses a
        proportion of the features' standard deviation as radius, while the latter uses a constant
        value.</br>
        2. The standard deviation proportion (value by which the standard deviation will be
        divided) or a constant radius value, as defined in the first position of the tuple.</br>
        3. [Optional] If the quantization policy is `'stddiv'`, an inicial radius value must be
        passed to be used as a cold-start for the feature quantization. If not provided, `0.01`
        is assumed as default.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.TrumpApproval()
    >>> model = tree.StochasticGradientTreeRegressor(grace_period=20)
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.828504

    Notes
    -----
    This implementation enhances the original proposal [^1] by using an incremental strategy to
    discretize numerical features dynamically, rather than relying on a calibration set and
    parameterized number of bins. The strategy used is an adaptation of the Quantile Observer
    (QO) [^2]. Different bin size setting policies are available for selection.
    They directly related to number of split candidates the tree is going to explore, and thus,
    how accurate its split decisions are going to be. Besides, the number of stored bins per
    feature is directly related to the tree's memory usage and runtime.

    References
    ---------
    [^1]: Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees.
    In Asian Conference on Machine Learning (pp. 1094-1109).
    [^2]: Mastelini, S.M. and de Carvalho, A.C.P.D.L.F., 2020. Using dynamical quantization to
    perform split attempts in online tree regressors. arXiv preprint arXiv:2012.00083.

    """

    def __init__(
        self,
        delta: float = 1e-7,
        grace_period: int = 200,
        init_pred: float = 0.0,
        max_depth: typing.Optional[int] = None,
        lambda_value: float = 0.1,
        gamma: float = 1.0,
        nominal_attributes: typing.Optional[typing.List] = None,
        quantization_strategy: typing.Tuple[str, float, typing.Optional[float]] = (
            "stddiv",
            3.0,
        ),
    ):

        super().__init__(
            delta=delta,
            grace_period=grace_period,
            init_pred=init_pred,
            max_depth=max_depth,
            lambda_value=lambda_value,
            gamma=gamma,
            nominal_attributes=nominal_attributes,
            quantization_strategy=quantization_strategy,
        )

        self._objective = SquaredErrorObjective()

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        return self._root.sort_instance_to_leaf(x).leaf_prediction()
