from __future__ import annotations

import abc
import math

from scipy.stats import f as f_dist

from river import base, tree

from .losses import BinaryCrossEntropyLoss, SquaredErrorLoss
from .nodes.branch import DTBranch, NominalMultiwayBranch, NumericBinaryBranch
from .nodes.sgt_nodes import SGTLeaf
from .utils import BranchFactory, GradHessMerit


class StochasticGradientTree(base.Estimator, abc.ABC):
    """Base Stochastic Gradient Tree (SGT) class.

    This class defines the main characteristics that are shared by the different SGT
    implementations.

    """

    def __init__(
        self,
        loss_func,
        delta,
        grace_period,
        init_pred,
        max_depth,
        lambda_value,
        gamma,
        nominal_attributes,
        feature_quantizer,
    ):
        # What really defines how a SGT works is its loss function
        self.loss_func = loss_func
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
        self.feature_quantizer = (
            feature_quantizer if feature_quantizer is not None else tree.splitter.StaticQuantizer()
        )

        self._root: SGTLeaf | DTBranch = SGTLeaf(prediction=self.init_pred)

        # set used to check whether categorical feature has been already split
        self._split_features = set()
        self._n_splits = 0
        self._n_node_updates = 0
        self._n_observations = 0

    def _target_transform(self, y):
        """Apply transformation to the raw target input.

        Different strategies are used for classification and regression. By default, use
        an identity function.

        Parameters
        ----------
        y
            The target value, over which the transformation will be applied.
        """
        return y

    def learn_one(self, x, y, *, w=1.0):
        self._n_observations += w

        """ Update Stochastic Gradient Tree with a single instance. """
        y_true_trs = self._target_transform(y)

        p_node = None
        node = None
        if not isinstance(self._root, SGTLeaf):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        # A leaf could not be reached in a single attempt let's deal with that
        if isinstance(node, NumericBinaryBranch) or isinstance(node, NominalMultiwayBranch):
            while True:
                # Split node encountered a previously unseen categorical value (in a multi-way
                #  test), so there is no branch to sort the instance to
                if node.max_branches() == -1 and node.feature in x:
                    # Create a new branch to the new categorical value
                    leaf = SGTLeaf(depth=node.depth + 1, split_params=node.stats.copy())
                    #
                    node.add_child(x[node.feature], leaf)
                    node = leaf
                # The split feature is missing in the instance. Hence, we pass the new example
                # to the most traversed path in the current subtree
                else:
                    _, node = node.most_common_path()
                    # And we keep trying to reach a leaf
                    if isinstance(node, DTBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, SGTLeaf):
                    break

            y_pred_raw = self.loss_func.transfer(node.prediction())
            grad_hess = self.loss_func.compute_derivatives(y_true_trs, y_pred_raw)
            node.update(x, grad_hess, self, w)
        else:  # Node is a leaf
            y_pred_raw = self.loss_func.transfer(node.prediction())
            grad_hess = self.loss_func.compute_derivatives(y_true_trs, y_pred_raw)
            node.update(x, grad_hess, self, w)

            if node.total_weight - node.last_split_attempt_at < self.grace_period:
                return self

            # Update split attempt data
            node.last_split_attempt_at = node.total_weight

            # If the maximum depth is reached, attempt to apply a "null split", i.e., update the
            # prediction value
            if node.depth >= self.max_depth:
                # Null split: update the prediction using the new gradient information
                best_split = BranchFactory()
                best_split.merit = GradHessMerit()
                best_split.merit.delta_pred = node.delta_prediction(
                    node.update_stats.mean, self.lambda_value
                )
                dlms = node.update_stats.delta_loss_mean_var(best_split.merit.delta_pred)
                best_split.merit.loss_mean = dlms.mean.get()
                best_split.merit.loss_var = dlms.get()
            else:  # Proceed with the standard split attempt procedure
                best_split = node.find_best_split(self)

            p = self._compute_p_value(best_split.merit, node.total_weight)

            if p < self.delta and best_split.merit.loss_mean < 0:
                p_branch = p_node.branch_no(x) if isinstance(p_node, DTBranch) else None
                node.apply_split(best_split, p_node, p_branch, self)

        return self

    @staticmethod
    def _compute_p_value(merit, n_observations):
        # Null hypothesis: expected loss is zero
        # Alternative hypothesis: expected loss is not zero

        f_value = (
            n_observations * (merit.loss_mean * merit.loss_mean) / merit.loss_var
            if merit.loss_var > 0.0
            else None
        )

        if f_value is None:
            return 1.0

        return 1 - f_dist.cdf(f_value, 1, n_observations - 1)

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
    def height(self) -> int:
        if self._root:
            return self._root.height
        return 0

    @property
    def n_nodes(self):
        if self._root:
            return self._root.n_nodes

    @property
    def n_branches(self):
        if self._root:
            return self._root.n_branches

    @property
    def n_leaves(self):
        if self._root:
            return self._root.n_leaves


class SGTClassifier(StochasticGradientTree, base.Classifier):
    """Stochastic Gradient Tree[^1] for binary classification.

    Binary decision tree classifier that minimizes the binary cross-entropy to guide its growth.

    Stochastic Gradient Trees (SGT) directly minimize a loss function to guide tree growth and
    update their predictions. Thus, they differ from other incrementally tree learners that do
    not directly optimize the loss, but data impurity-related heuristics.

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
    feature_quantizer
        The algorithm used to quantize numeric features. Either a static quantizer (as in the
        original implementation) or a dynamic quantizer can be used. The correct choice and setup
        of the feature quantizer is a crucial step to determine the performance of SGTs.
        Feature quantizers are akin to the attribute observers used in Hoeffding Trees. By
        default, an instance of `tree.splitter.StaticQuantizer` (with default parameters) is
        used if this parameter is not set.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Phishing()
    >>> model = tree.SGTClassifier(
    ...     feature_quantizer=tree.splitter.StaticQuantizer(
    ...         n_bins=32, warm_start=10
    ...     )
    ... )
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 82.24%

    References
    ----------
    [^1]: Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees.
    In Asian Conference on Machine Learning (pp. 1094-1109).

    """

    def __init__(
        self,
        delta: float = 1e-7,
        grace_period: int = 200,
        init_pred: float = 0.0,
        max_depth: int | None = None,
        lambda_value: float = 0.1,
        gamma: float = 1.0,
        nominal_attributes: list | None = None,
        feature_quantizer: tree.splitter.Quantizer | None = None,
    ):
        super().__init__(
            loss_func=BinaryCrossEntropyLoss(),
            delta=delta,
            grace_period=grace_period,
            init_pred=init_pred,
            max_depth=max_depth,
            lambda_value=lambda_value,
            gamma=gamma,
            nominal_attributes=nominal_attributes,
            feature_quantizer=feature_quantizer,
        )

    def _target_transform(self, y):
        return float(y)

    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        if isinstance(self._root, DTBranch):
            leaf = self._root.traverse(x, until_leaf=True)
        else:
            leaf = self._root

        t_proba = self.loss_func.transfer(leaf.prediction())  # type: ignore

        return {True: t_proba, False: 1 - t_proba}


class SGTRegressor(StochasticGradientTree, base.Regressor):
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
    feature_quantizer
        The algorithm used to quantize numeric features. Either a static quantizer (as in the
        original implementation) or a dynamic quantizer can be used. The correct choice and setup
        of the feature quantizer is a crucial step to determine the performance of SGTs.
        Feature quantizers are akin to the attribute observers used in Hoeffding Trees. By
        default, an instance of `tree.splitter.StaticQuantizer` (with default parameters) is
        used if this parameter is not set.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.TrumpApproval()
    >>> model = tree.SGTRegressor(
    ...     delta=0.01,
    ...     lambda_value=0.01,
    ...     grace_period=20,
    ...     feature_quantizer=tree.splitter.DynamicQuantizer(std_prop=0.1)
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.721818

    Notes
    -----
    This implementation enhances the original proposal [^1] by using an incremental strategy to
    discretize numerical features dynamically, rather than relying on a calibration set and
    parameterized number of bins. The strategy used is an adaptation of the Quantization Observer
    (QO) [^2]. Different bin size setting policies are available for selection.
    They directly related to number of split candidates the tree is going to explore, and thus,
    how accurate its split decisions are going to be. Besides, the number of stored bins per
    feature is directly related to the tree's memory usage and runtime.

    References
    ----------
    [^1]: Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees.
    In Asian Conference on Machine Learning (pp. 1094-1109).
    [^2]: Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization
    to perform split attempts in online tree regressors. Pattern Recognition Letters.

    """

    def __init__(
        self,
        delta: float = 1e-7,
        grace_period: int = 200,
        init_pred: float = 0.0,
        max_depth: int | None = None,
        lambda_value: float = 0.1,
        gamma: float = 1.0,
        nominal_attributes: list | None = None,
        feature_quantizer: tree.splitter.Quantizer | None = None,
    ):
        super().__init__(
            loss_func=SquaredErrorLoss(),
            delta=delta,
            grace_period=grace_period,
            init_pred=init_pred,
            max_depth=max_depth,
            lambda_value=lambda_value,
            gamma=gamma,
            nominal_attributes=nominal_attributes,
            feature_quantizer=feature_quantizer,
        )

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if isinstance(self._root, DTBranch):
            leaf = self._root.traverse(x, until_leaf=True)
        else:
            leaf = self._root
        return self.loss_func.transfer(leaf.prediction())  # type: ignore
