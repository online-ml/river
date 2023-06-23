from __future__ import annotations

import math

from river import stats

from ..utils import GradHess, GradHessStats
from .base import Quantizer


class DynamicQuantizer(Quantizer):
    """Adapted version of the Quantizer Observer (QO)[^1] that is applied to Stochastic Gradient
    Trees (SGT).

    This feature quantizer starts by partitioning the inputs using the passed `radius` value.
    As more splits are created in the SGTs, new feature quantizers will use `std * std_prop` as
    the quantization radius. In the expression, `std` represents the standard deviation of the
    input data, which is calculated incrementally.

    Parameters
    ----------
    radius
        The initial quantization radius.
    std_prop
        The proportion of the standard deviation that is going to be used to define the radius
        value for new quantizer instances following the initial one.

    References
    ----------
    [^1]: Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization
    to perform split attempts in online tree regressors. Pattern Recognition Letters.
    """

    def __init__(self, radius: float = 0.5, std_prop: float = 0.25):
        super().__init__()
        self.radius = radius
        self.std_prop = std_prop

        self.feat_var = stats.Var()
        self.hash: dict[int, GradHessStats] = {}

    def update(self, x_val, gh: GradHess, w: float):
        self.feat_var.update(x_val, w)

        index = math.floor(x_val / self.radius)
        if index in self.hash:
            self.hash[index].update(gh, w)
        else:
            ghs = GradHessStats()
            ghs.update(gh, w)
            self.hash[index] = ghs

    def __len__(self):
        return len(self.hash)

    def __getitem__(self, k):
        return self.hash[k]

    def __iter__(self):
        for k in sorted(self.hash):
            yield self.radius * (k + 1), self.hash[k]

    def _get_params(self):
        params = super()._get_params()
        new_radius = self.std_prop * math.sqrt(self.feat_var.get())

        if new_radius > 0:
            params["radius"] = new_radius

        return params


class StaticQuantizer(Quantizer):
    """Quantization strategy originally used in Stochastic Gradient Trees (SGT)[^1].

    Firstly, a buffer of size `warm_start` is stored. The data stored in the buffer is then used
    to quantize the input feature into `n_bins` intervals. These intervals will be replicated
    to every new quantizer. Feature values lying outside of the limits defined by the initial
    buffer will be mapped to the head or tail of the list of intervals.

    Parameters
    ----------
    n_bins
        The number of bins (intervals) to divide the input feature.
    warm_start
        The number of observations used to initialize the quantization intervals.
    buckets
        This parameter is only used internally by the quantizer, so it must not be set.
        Once the intervals are defined, new instances of this quantizer will receive the
        quantization information via this parameter.

    References
    ----------
    [^1]: Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees.
    In Asian Conference on Machine Learning (pp. 1094-1109).

    """

    def __init__(self, n_bins: int = 64, warm_start: int = 100, *, buckets: list | None = None):
        super().__init__()

        self.n_bins = n_bins
        self.warm_start = warm_start
        self.buckets = buckets

        if self.buckets is None:
            self._buffer: list[tuple] = []
            self._min = None
            self._radius = None
        else:
            self._buffer = None  # type: ignore
            # Define the quantization radius and the minimum value to data perform shift
            self._radius = self.buckets[1][0][1] - self.buckets[1][0][0]
            self._min = self.buckets[0][0][1] - self._radius

    def update(self, x_val, gh: GradHess, w: float):
        if self.buckets is None:
            self._buffer.append((x_val, gh, w))

            if len(self._buffer) < self.warm_start:
                return

            self._min = min(self._buffer, key=lambda t: t[0])[0]
            _max = max(self._buffer, key=lambda t: t[0])[0]
            self._radius = (_max - self._min) / self.n_bins

            splits = (
                [-math.inf]
                + [self._min + i * self._radius for i in range(1, self.n_bins)]
                + [math.inf]
            )

            self.buckets = [
                ((splits[i], splits[i + 1]), GradHessStats()) for i in range(self.n_bins)
            ]

            # Replay buffer
            for x_val, gh, w in self._buffer:
                pos = math.floor((x_val - self._min) / self._radius)
                if pos >= self.n_bins:
                    pos = self.n_bins - 1
                self.buckets[pos][1].update(gh, w)

            # And empty it
            self._buffer = None  # type: ignore

        # Projection scheme
        pos = math.floor((x_val - self._min) / self._radius)
        if pos < 0:
            pos = 0
        if pos >= self.n_bins:
            pos = self.n_bins - 1

        # Update the corresponding bucket
        self.buckets[pos][1].update(gh, w)

    def __len__(self):
        if self.buckets:
            return len(self.buckets)
        return 0

    def __iter__(self):
        if self.buckets is None:
            return

        for x_range, ghs in self.buckets:
            if ghs.total_weight == 0:
                continue
            yield x_range[1], ghs

    def _get_params(self):
        params = super()._get_params()

        if self.buckets is not None:
            # Create buckets with empty stats: only the tuples with data range are kept
            buckets = [(b[0], GradHessStats()) for b in self.buckets]

            params["buckets"] = buckets
        return params
