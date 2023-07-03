from __future__ import annotations

import random
import statistics
from copy import deepcopy

from river import base, drift

from .hoeffding_tree_regressor import HoeffdingTreeRegressor
from .nodes.branch import DTBranch
from .nodes.hatr_nodes import (
    AdaBranchRegressor,
    AdaLeafRegAdaptive,
    AdaLeafRegMean,
    AdaLeafRegModel,
    AdaLeafRegressor,
    AdaNomBinaryBranchReg,
    AdaNomMultiwayBranchReg,
    AdaNumBinaryBranchReg,
    AdaNumMultiwayBranchReg,
)
from .splitter import Splitter


class HoeffdingAdaptiveTreeRegressor(HoeffdingTreeRegressor):
    """Hoeffding Adaptive Tree regressor (HATR).

    This class implements a regression version of the Hoeffding Adaptive Tree Classifier. Hence,
    it also uses an ADWIN concept-drift detector instance at each decision node to monitor
    possible changes in the data distribution. If a drift is detected in a node, an alternate
    tree begins to be induced in the background. When enough information is gathered, HATR
    swaps the node where the change was detected by its alternate tree.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
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
        List of Nominal attributes. If empty, then assume that all numeric attributes should
        be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
    min_samples_split
        The minimum number of samples every branch resulting from a split candidate must have
        to be considered valid.
    bootstrap_sampling
        If True, perform bootstrap sampling in the leaf nodes.
    drift_window_threshold
        Minimum number of examples an alternate tree must observe before being considered as a
        potential replacement to the current one.
    drift_detector
        The drift detector used to build the tree. If `None` then `drift.ADWIN` is used.
        Only detectors that support arbitrarily valued continuous data can be used for regression.
    switch_significance
        The significance level to assess whether alternate subtrees are significantly better
        than their main subtree counterparts.
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
    seed
       Random seed for reproducibility.

    Notes
    -----
    The Hoeffding Adaptive Tree [^1] uses drift detectors to monitor performance of branches
    in the tree and to replace them with new branches when their accuracy decreases.

    The bootstrap sampling strategy is an improvement over the original Hoeffding Adaptive Tree
    algorithm. It is enabled by default since, in general, it results in better performance.

    To cope with ADWIN's requirements of bounded input data, HATR uses a novel error normalization
    strategy based on the empiral rule of Gaussian distributions. We assume the deviations
    of the predictions from the expected values follow a normal distribution. Hence, we subject
    these errors to a min-max normalization assuming that most of the data lies in the
    $\\left[-3\\sigma, 3\\sigma\\right]$ range. These normalized errors are passed to the ADWIN
    instances. This is the same strategy used by Adaptive Random Forest Regressor.

    References
    ----------
    [^1]: Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data streams."
    In International Symposium on Intelligent Data Analysis, pp. 249-260. Springer, Berlin,
    Heidelberg, 2009.

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
    ...     tree.HoeffdingAdaptiveTreeRegressor(
    ...         grace_period=50,
    ...         model_selector_decay=0.3,
    ...         seed=0
    ...     )
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.823026

    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int | None = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        min_samples_split: int = 5,
        bootstrap_sampling: bool = True,
        drift_window_threshold: int = 300,
        drift_detector: base.DriftDetector | None = None,
        switch_significance: float = 0.05,
        binary_split: bool = False,
        max_size: float = 500.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            delta=delta,
            tau=tau,
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

        self.bootstrap_sampling = bootstrap_sampling
        self.drift_window_threshold = drift_window_threshold
        self.drift_detector = drift_detector if drift_detector is not None else drift.ADWIN()
        self.switch_significance = switch_significance
        self.seed = seed

        self._n_alternate_trees = 0
        self._n_pruned_alternate_trees = 0
        self._n_switch_alternate_trees = 0
        self._norm_dist = statistics.NormalDist()

        self._rng = random.Random(self.seed)

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau", "drift_window_threshold", "switch_significance"}

    @property
    def n_alternate_trees(self):
        return self._n_alternate_trees

    @property
    def n_pruned_alternate_trees(self):
        return self._n_pruned_alternate_trees

    @property
    def n_switch_alternate_trees(self):
        return self._n_switch_alternate_trees

    @property
    def summary(self):
        summ = super().summary
        summ.update(
            {
                "n_alternate_trees": self.n_alternate_trees,
                "n_pruned_alternate_trees": self.n_pruned_alternate_trees,
                "n_switch_alternate_trees": self.n_switch_alternate_trees,
            }
        )
        return summ

    def learn_one(self, x, y, *, sample_weight=1.0):
        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1
        self._root.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    def predict_one(self, x):
        pred = 0.0
        if self._root is not None:
            found_nodes = [self._root]
            if isinstance(self._root, DTBranch):
                found_nodes = self._root.traverse(x, until_leaf=True)
            for leaf in found_nodes:
                pred += leaf.prediction(x, tree=self)
            # Mean prediction among the reached leaves
            pred /= len(found_nodes)

        return pred

    def _new_leaf(self, initial_stats=None, parent=None, is_active=True):
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """
        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        leaf_model = None
        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_model = deepcopy(self.leaf_model)
            else:
                try:
                    leaf_model = deepcopy(parent._leaf_model)  # noqa
                except AttributeError:
                    leaf_model = deepcopy(self.leaf_model)

        if self.leaf_prediction == self._TARGET_MEAN:
            return AdaLeafRegMean(
                initial_stats,
                depth,
                self.splitter,
                drift_detector=self.drift_detector.clone(),
                rng=self._rng,
            )
        elif self.leaf_prediction == self._MODEL:
            return AdaLeafRegModel(
                initial_stats,
                depth,
                self.splitter,
                drift_detector=self.drift_detector.clone(),
                rng=self._rng,
                leaf_model=leaf_model,
            )
        else:  # adaptive learning node
            new_adaptive = AdaLeafRegAdaptive(
                initial_stats,
                depth,
                self.splitter,
                drift_detector=self.drift_detector.clone(),
                rng=self._rng,
                leaf_model=leaf_model,
            )
            if parent is not None and isinstance(parent, AdaLeafRegressor):
                new_adaptive._fmse_mean = parent._fmse_mean  # noqa
                new_adaptive._fmse_model = parent._fmse_model  # noqa

            return new_adaptive

    def _branch_selector(
        self, numerical_feature=True, multiway_split=False
    ) -> type[AdaBranchRegressor]:
        """Create a new split node."""
        if numerical_feature:
            if not multiway_split:
                return AdaNumBinaryBranchReg
            else:
                return AdaNumMultiwayBranchReg
        else:
            if not multiway_split:
                return AdaNomBinaryBranchReg
            else:
                return AdaNomMultiwayBranchReg

    @classmethod
    def _unit_test_params(cls):
        yield {"seed": 1}
