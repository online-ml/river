from __future__ import annotations

import functools
import math
import typing

from river import stats

from ..utils import BranchFactory
from .base import Splitter

if typing.TYPE_CHECKING:
    from river import utils


class QOSplitter(Splitter):
    """Quantization observer (QO).

    This splitter utilizes a hash-based quantization algorithm to keep track of the target
    statistics and evaluate split candidates. QO, relies on the radius parameter to define
    discretization intervals for each incoming feature. Split candidates are defined as the
    midpoints between two consecutive hash slots. Both binary splits and multi-way splits can be
    created by this attribute observer. This class implements the algorithm described
    in [^1].

    The smaller the quantization radius, the more hash slots will be created to accommodate the
    discretized data. Hence, both the running time and memory consumption increase, but the
    resulting splits ought to be closer to the ones obtained by a batch exhaustive approach.
    On the other hand, if the radius is too large, fewer slots will be created, less memory and
    running time will be required, but at the cost of coarse split suggestions.

    QO assumes that all features have the same range. It is always advised to scale the features
    to apply this splitter. That can be done using the `preprocessing` module. A good "rule of
    thumb" is to scale data using `preprocessing.StandardScaler` and define the radius as a
    proportion of the features' standard deviation. For instance, the default radius value
    would correspond to one quarter of the normalized features' standard deviation (since the
    scaled data has zero mean and unit variance). If the features come from normal
    distributions, by following the empirical rule, roughly `32` hash slots will be created.

    Parameters
    ----------
    radius
        The quantization radius. QO discretizes the incoming feature in intervals of equal length
        that are defined by this parameter.
    allow_multiway_splits
        Whether or not allow that multiway splits are evaluated. Numeric multi-way splits use
        the same quantization strategy of QO to create multiple tree branches. The same
        quantization radius is used, and each stored slot represents the split enabling
        statistics of one branch.


    References
    ----------
    [^1]: Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization to
    perform split attempts in online tree regressors. Pattern Recognition Letters.

    """

    def __init__(self, radius: float = 0.25, allow_multiway_splits=False):
        super().__init__()

        if radius <= 0:
            raise ValueError("'radius' must be greater than zero.")
        self.radius = radius
        self._quantizer = FeatureQuantizer(radius=self.radius)

        self.allow_multiway_splits = allow_multiway_splits

    def update(self, att_val, target_val, sample_weight):
        if att_val is None:
            return
        else:
            self._quantizer.update(att_val, target_val, sample_weight)

    def cond_proba(self, att_val, target_val):
        raise NotImplementedError

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only=True):
        candidate = BranchFactory()

        # In case the hash carries just one element return the null split
        if len(self._quantizer) == 1:
            return candidate

        # Numeric multiway test
        if self.allow_multiway_splits and not binary_only:
            slot_ids, post_split_dists = self._quantizer.all()
            merit = criterion.merit_of_split(pre_split_dist, post_split_dists)

            candidate = BranchFactory(
                merit,
                att_idx,
                (self.radius, slot_ids),
                post_split_dists,
                numerical_feature=True,
                multiway_split=True,
            )

        # The previously evaluated x value
        prev_x = None

        for x, left_dist in self._quantizer:
            # First hash element
            if prev_x is None:
                prev_x = x
                continue

            right_dist = pre_split_dist - left_dist
            post_split_dists = [left_dist, right_dist]
            merit = criterion.merit_of_split(pre_split_dist, post_split_dists)

            if merit > candidate.merit:
                split_point = (prev_x + x) / 2.0
                candidate = BranchFactory(
                    merit,
                    att_idx,
                    split_point,
                    post_split_dists,
                    numerical_feature=True,
                    multiway_split=False,
                )

            prev_x = x
        return candidate

    @property
    def is_target_class(self) -> bool:
        return False


class Slot:
    """The element stored in the quantization hash.

    Each slot keeps the mean values of the numerical feature, as well as the variance
    and mean of the target.

    """

    def __init__(
        self,
        x: float,
        y=typing.Union[float, "utils.VectorDict"],
        weight: float = 1.0,
    ):
        self.x_stats = stats.Mean()
        self.x_stats.update(x, weight)

        self.y_stats: stats.Var | utils.VectorDict

        self._update_estimator: typing.Callable[[float | utils.VectorDict, float], None]
        self.is_single_target = True

        self._init_estimator(y)
        self._update_estimator(y, weight)

    def _init_estimator(self, y):
        if isinstance(y, dict):
            self.is_single_target = False
            self.y_stats = utils.VectorDict(default_factory=functools.partial(stats.Var))
            self._update_estimator = self._update_estimator_multivariate
        else:
            self.y_stats = stats.Var()
            self._update_estimator = self._update_estimator_univariate

    def _update_estimator_univariate(self, target, sample_weight):
        self.y_stats.update(target, sample_weight)

    def _update_estimator_multivariate(self, target, sample_weight):
        for t in target:
            self.y_stats[t].update(target[t], sample_weight)

    def __iadd__(self, o):
        self.x_stats += o.x_stats
        self.y_stats += o.y_stats

        return self

    def update(self, x, y, sample_weight):
        self.x_stats.update(x, sample_weight)
        self._update_estimator(y, sample_weight)


class FeatureQuantizer:
    """The actual dynamic feature quantization mechanism.

    The input numerical feature is partitioned using equal-sized intervals that are defined
    by the `radius` parameter. For each interval, estimators of target's mean and variance are
    kept, as well as an estimator of the feature's mean value. At any time, one can iterate
    over the ordered stored values in the hash. The ordering is defined by the quantized values
    of the feature.

    """

    def __init__(self, radius: float):
        self.radius = radius
        self.hash: dict[int, Slot] = {}

    def __getitem__(self, k):
        return self.hash[k]

    def __len__(self):
        return len(self.hash)

    def update(self, x: float, y: float | utils.VectorDict, weight: float):
        index = math.floor(x / self.radius)
        try:
            self.hash[index].update(x, y, weight)
        except KeyError:
            self.hash[index] = Slot(x, y, weight)

    def __iter__(self):
        # Import river.utils here to prevent circular import of river.utils
        from river import utils

        aux_stats = (
            stats.Var()
            if next(iter(self.hash.values())).is_single_target
            else utils.VectorDict(default_factory=functools.partial(stats.Var))
        )

        for i in sorted(self.hash.keys()):
            x = self.hash[i].x_stats.get()
            aux_stats += self.hash[i].y_stats
            yield x, aux_stats

    def all(self):
        branch_ids = []
        split_stats = []
        for slot_id in sorted(self.hash.keys()):
            branch_ids.append(slot_id)
            split_stats.append(self.hash[slot_id].y_stats)

        return branch_ids, split_stats
