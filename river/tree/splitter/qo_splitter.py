import functools
import math
import typing

from river.stats import Mean, Var
from river.utils import VectorDict

from .._attribute_test import AttributeSplitSuggestion, NumericAttributeBinaryTest
from .base_splitter import Splitter


class QOSplitter(Splitter):
    """Quantization observer (QO).

    Utilizes a dynamical hash-based quantization algorithm to keep track of the target statistics
    and evaluate split candidates. This class implements the algorithm described in [^1].
    This attribute observer keeps an internal estimator of the input feature's variance. By doing
    that, QO can calculate better values for its radius parameter to be passed to future learning
    nodes.

    Parameters
    ----------
    radius
        The quantization radius.

    References
    ----------
    [^1]: Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization to
    perform split attempts in online tree regressors. Pattern Recognition Letters.

    """

    def __init__(self, radius: float = 0.01):
        super().__init__()
        self.radius = radius if radius > 0 else 0.01
        self._x_var = Var()
        self._quantizer = FeatureQuantizer(radius=self.radius)

    def update(self, att_val, target_val, sample_weight):
        if att_val is None:
            return
        else:
            self._x_var.update(att_val, sample_weight)
            self._quantizer.update(att_val, target_val, sample_weight)

    def cond_proba(self, att_val, class_val):
        raise NotImplementedError

    def best_evaluated_split_suggestion(
        self, criterion, pre_split_dist, att_idx, binary_only=True
    ):
        candidate = AttributeSplitSuggestion(None, [{}], -math.inf)

        # The previously evaluated x value
        prev_x = None

        for (x, left_dist) in self._quantizer:
            # First hash element
            if prev_x is None:
                # In case the hash carries just one element return the null split
                if len(self._quantizer) == 1:
                    return candidate
                prev_x = x
                continue

            right_dist = pre_split_dist - left_dist
            post_split_dists = [left_dist, right_dist]
            merit = criterion.merit_of_split(pre_split_dist, post_split_dists)

            if merit > candidate.merit:
                split_point = (prev_x + x) / 2.0
                candidate = self._update_candidate(
                    split_point, att_idx, post_split_dists, merit
                )

            prev_x = x
        return candidate

    @property
    def x_std(self) -> float:
        """The standard deviation of the monitored feature."""
        return math.sqrt(self._x_var.get())

    @staticmethod
    def _update_candidate(split_point, att_idx, post_split_dists, merit):
        num_att_binary_test = NumericAttributeBinaryTest(att_idx, split_point, True)
        candidate = AttributeSplitSuggestion(
            num_att_binary_test, post_split_dists, merit
        )

        return candidate


class Slot:
    """ The element stored in the quantization hash.

    Each slot keeps the mean values of the numerical feature, as well as the variance
    and mean of the target.

    """

    def __init__(
        self, x: float, y=typing.Union[float, VectorDict], weight: float = 1.0
    ):
        self.x_stats = Mean()
        self.x_stats.update(x, weight)

        self.y_stats: typing.Union[Var, VectorDict]

        self._update_estimator: typing.Callable[
            [typing.Union[float, VectorDict], float], None
        ]
        self.is_single_target = True

        self._init_estimator(y)
        self._update_estimator(y, weight)

    def _init_estimator(self, y):
        if isinstance(y, dict):
            self.is_single_target = False
            self.y_stats = VectorDict(default_factory=functools.partial(Var))
            self._update_estimator = self._update_estimator_multivariate
        else:
            self.y_stats = Var()
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
    """ The actual dynamic feature quantization mechanism.

    The input numerical feature is partitioned using equal-sized intervals that are defined
    by the `radius` parameter. For each interval, estimators of target's mean and variance are
    kept, as well as an estimator of the feature's mean value. At any time, one can iterate
    over the ordered stored values in the hash. The ordering is defined by the quantized values
    of the feature.

    """

    def __init__(self, radius: float):
        self.radius = radius
        self.hash = {}

    def __getitem__(self, k):
        return self.hash[k]

    def __len__(self):
        return len(self.hash)

    def update(self, x: float, y: typing.Union[float, VectorDict], weight: float):
        index = math.floor(x / self.radius)
        try:
            self.hash[index].update(x, y, weight)
        except KeyError:
            self.hash[index] = Slot(x, y, weight)

    def __iter__(self):
        aux_stats = (
            Var()
            if next(iter(self.hash.values())).is_single_target
            else VectorDict(default_factory=functools.partial(Var))
        )

        for i in sorted(self.hash.keys()):
            x = self.hash[i].x_stats.get()
            aux_stats += self.hash[i].y_stats
            yield x, aux_stats
