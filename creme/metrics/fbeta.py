import collections
import itertools
import functools
import statistics

from . import base
from . import confusion
from . import precision
from . import recall


__all__ = [
    'F1',
    'FBeta',
    'MacroF1',
    'MacroFBeta',
    'MicroF1',
    'MicroFBeta',
    'MultiFBeta',
    'RollingF1',
    'RollingFBeta',
    'RollingMacroF1',
    'RollingMacroFBeta',
    'RollingMicroF1',
    'RollingMicroFBeta',
    'RollingMultiFBeta'
]


class BaseFBeta:

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class FBeta(BaseFBeta, base.BinaryMetric):
    """Binary F-Beta score.

    The FBeta score is a weighted harmonic mean between precision and recall. The higher the
    ``beta`` value, the higher the recall will be taken into account. When ``beta`` equals 1,
    precision and recall and equivalently weighted, which results in the F1 score (see
    `metrics.F1`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.FBeta(beta=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            FBeta: 0.357143

    """

    def __init__(self, beta):

        if beta <= 0:
            raise ValueError('beta should be strictly positive')

        self.beta = beta
        self.precision = precision.Precision()
        self.recall = recall.Recall()

    def update(self, y_true, y_pred):
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        return self

    def get(self):
        p = self.precision.get()
        r = self.recall.get()
        b2 = self.beta ** 2
        try:
            return (1 + b2) * p * r / (b2 * p + r)
        except ZeroDivisionError:
            return 0.0


class MacroFBeta(BaseFBeta, base.MultiClassMetric):
    """Macro-average F-Beta score.

    This works by computing the F-Beta score per class, and then performs a global average.

    Parameters:
        beta (float): Weight of precision in harmonic mean.
        window_size (int): Size of the rolling window.

    Attributes:
        fbetas (collections.defaultdict): F-Beta scores per class.
        classes (set): Observed classes.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroFBeta(beta=0.8)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MacroFBeta: 1.
            MacroFBeta: 0.310606
            MacroFBeta: 0.540404
            MacroFBeta: 0.540404
            MacroFBeta: 0.485982

    """

    def __init__(self, beta):
        self.fbetas = collections.defaultdict(functools.partial(FBeta, beta=beta))
        self.classes = set()

    def update(self, y_true, y_pred):
        self.classes.update({y_true, y_pred})

        for c in self.classes:
            self.fbetas[c].update(y_true == c, y_pred == c)

        return self

    def get(self):
        total = sum(fbeta.get() for fbeta in self.fbetas.values())
        try:
            return total / len(self.fbetas)
        except ZeroDivisionError:
            return 0.


class MicroFBeta(FBeta, base.MultiClassMetric):
    """Micro-average F-Beta score.

    This computes the F-Beta score by merging all the predictions and true labels, and then
    computes a global F-Beta score.

    Parameters:
        beta (float): Weight of precision in harmonic mean.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0]
            >>> y_pred = [0, 1, 1, 2, 1]

            >>> metric = metrics.MicroFBeta(beta=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            MicroFBeta: 0.6

    """

    def __init__(self, beta):
        super().__init__(beta)
        self.precision = precision.MicroPrecision()
        self.recall = recall.MicroRecall()


class MultiFBeta(BaseFBeta, base.MultiClassMetric):
    """Multi-class F-Beta score with different betas per class.

    The multiclass F-Beta score is the arithmetic average of the binary F-Beta scores of each class.
    The mean can be weighted by providing class weights.

    Parameters:
        beta (dict): Weight of precision in harmonic mean. For different beta values per
            class, provide a dict with class labels as keys and beta values as values.
        weights (dict): Class weights. If not provided then uniform weights will be used

    Attributes:
        classes (set): Metric different classes.
        fbetas (collections.defaultdict): Classes fbeta values.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MultiFBeta(
            ...     betas={0: 0.25, 1: 1, 2: 4},
            ...     weights={0: 1, 1: 1, 2: 2}
            ... )

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MultiFBeta: 1.
            MultiFBeta: 0.257576
            MultiFBeta: 0.628788
            MultiFBeta: 0.628788
            MultiFBeta: 0.468788

    """

    def __init__(self, betas, weights=None):
        self.betas = betas
        self.fbetas = dict()
        self.classes = set()
        self.weights = (
            weights
            if weights is not None
            else collections.defaultdict(functools.partial(int, 1))
        )

    def update(self, y_true, y_pred):
        self.classes.update({y_true, y_pred})

        for c in self.classes:
            try:
                fb = self.fbetas[c]
            except KeyError:
                fb = self.fbetas[c] = FBeta(beta=self.betas[c])

            fb.update(y_true == c, y_pred == c)

        return self

    def get(self):
        total = sum(fb.get() * self.weights[label] for label, fb in self.fbetas.items())
        denom = sum(self.weights[label] for label in self.fbetas.keys())
        try:
            return total / denom
        except ZeroDivisionError:
            return 0.


class RollingFBeta(FBeta):
    """Rolling binary F-Beta score.

    The FBeta score is a weighted harmonic mean between precision and recall. The higher the
    ``beta`` value, the higher the recall will be taken into account. When ``beta`` equals 1,
    precision and recall and equivalently weighted, which results in the F1 score (see
    `metrics.RollingF1`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.
        window_size (int): Size of the rolling window.

    Attributes:
        precision (metrics.RollingPrecision): Rolling precision score.
        recall (metrics.RollingRecall): Rolling recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.RollingFBeta(beta=2, window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            0.0
            0.0
            0.0
            0.8333333333333334
            0.5
            0.3846153846153846

    """

    def __init__(self, beta, window_size):
        super().__init__(beta)
        self.precision = precision.RollingPrecision(window_size=window_size)
        self.recall = recall.RollingRecall(window_size=window_size)
        self.window_size = window_size


class RollingMacroFBeta(MacroFBeta):
    """Rolling macro-average F-Beta score.

    This works by computing the F-Beta score per class, and then performs a global average.

    Parameters:
        beta (float): Weight of precision in harmonic mean.
        window_size (int): Size of the rolling window.

    Attributes:
        fbetas (collections.defaultdict): Rolling F-Beta scores per class.
        classes (set): Observed classes.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMacroFBeta(beta=0.8, window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMacroFBeta: 1.
            RollingMacroFBeta: 0.310606
            RollingMacroFBeta: 0.540404
            RollingMacroFBeta: 0.333333
            RollingMacroFBeta: 0.418367

    """

    def __init__(self, beta, window_size):
        self.beta = beta
        self.window_size = window_size
        self.rcm = confusion.RollingConfusionMatrix(window_size=window_size)

    def update(self, y_true, y_pred):
        self.rcm.update(y_true, y_pred)
        return self

    def get(self):

        # Use the rolling confusion matric to count the TPs, FPs, and FNs
        classes = self.rcm.classes
        tps = collections.defaultdict(int)
        fps = collections.defaultdict(int)
        fns = collections.defaultdict(int)

        for yt, yp in itertools.product(classes, repeat=2):
            if yt == yp:
                tps[yp] = self.rcm.get(yt, {}).get(yp, 0)
            else:
                fps[yp] += self.rcm.get(yt, {}).get(yp, 0)
                fns[yp] += self.rcm.get(yp, {}).get(yt, 0)

        def div_or_0(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0.

        # Compute precisions
        ps = {c: div_or_0(tps[c], tps[c] + fps[c]) for c in classes}
        # Compute recalls
        rs = {c: div_or_0(tps[c], tps[c] + fns[c]) for c in classes}
        # Compute F-Beta scores
        b2 = self.beta ** 2
        fbs = {c: div_or_0((1 + b2) * ps[c] * rs[c], b2 * ps[c] + rs[c]) for c in classes}

        return statistics.mean(fbs.values())


class RollingMicroFBeta(RollingFBeta, base.MultiClassMetric):
    """Rolling micro multiclass FBeta score.

    The Micro FBeta score is a specific case of the FBeta score where the metric is globally
    computed by counting the total true positives, false negatives and false positives.
    ``beta`` parameter allows to weight recall higher than precision if sets to more than 1 or
    lower if sets between 0 and 1. For ``beta`` equals to 1, precision and recall and equivalently
    weighted resulting to traditional micro F1 score (see `metrics.MicroF1`).

    Parameters:
        beta (float): Weight of precision in harmonic mean.
        window_size (int): Size of the rolling window.

    Attributes:
        beta2 (float): Beta squared value.
        precision (metric.RollingPrecision): Rolling precision score.
        recall (metric.RollingRecall): Rolling recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0]
            >>> y_pred = [0, 1, 1, 2, 1]

            >>> metric = metrics.RollingMicroFBeta(beta=2, window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            1.0
            1.0
            0.6666666666666666
            0.6666666666666666
            0.3333333333333333

    """

    def __init__(self, beta, window_size):
        super().__init__(beta, window_size)
        self.precision = precision.RollingMicroPrecision(window_size=window_size)
        self.recall = recall.RollingMicroRecall(window_size=window_size)



class RollingMultiFBeta(MultiFBeta):
    """Rolling multi-class F-Beta score with different betas per class.

    The multiclass F-Beta score is the arithmetic average of the binary F-Beta scores of each
    class. The mean can be weighted by providing class weights.

    Parameters:
        beta (dict): Weight of precision in harmonic mean. For different beta values per
            class, provide a dict with class labels as keys and beta values as values.
        weights (dict): Class weights. If not provided then uniform weights will be used
        window_size (int): Size of the rolling window.

    Attributes:
        classes (set): Metric different classes.
        fbetas (collections.defaultdict): Class F-Beta scores.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0, 1]
            >>> y_pred = [0, 0, 2, 2, 1, 1]

            >>> metric = metrics.RollingMultiFBeta(betas={0: 0.25, 1: 1, 2: 4}, window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMultiFBeta: 1.
            RollingMultiFBeta: 0.257576
            RollingMultiFBeta: 0.505051
            RollingMultiFBeta: 0.333333
            RollingMultiFBeta: 0.333333
            RollingMultiFBeta: 0.555556

    """

    def __init__(self, betas, window_size, weights=None):
        super().__init__(betas=betas, weights=weights)
        self.window_size = window_size

    def update(self, y_true, y_pred):
        self.classes.update({y_true, y_pred})
        for c in self.classes:
            try:
                fb = self.fbetas[c]
            except KeyError:
                fb = self.fbetas[c] = RollingFBeta(beta=self.betas[c], window_size=self.window_size)

            fb.update(y_true == c, y_pred == c)

        return self


class F1(FBeta):
    """Binary F1 score.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.F1()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            F1: 0.4

    """

    def __init__(self):
        super().__init__(beta=1.)


class MacroF1(MacroFBeta):
    """Macro-average F1 score.

    This works by computing the F1 score per class, and then performs a global average.

    Parameters:
        window_size (int): Size of the rolling window.

    Attributes:
        fbetas (collections.defaultdict): F-Beta scores per class.
        classes (set): Observed classes.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroF1()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MacroF1: 1.
            MacroF1: 0.333333
            MacroF1: 0.555556
            MacroF1: 0.555556
            MacroF1: 0.488889

    """

    def __init__(self):
        super().__init__(beta=1.)


class MicroF1(MicroFBeta):
    """Micro-average F1 score.

    This computes the F1 score by merging all the predictions and true labels, and then computes a
    global F1 score.

    Attributes:
        precision (metrics.Precision): Precision score.
        recall (metrics.Recall): Recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0]
            >>> y_pred = [0, 1, 1, 2, 1]

            >>> metric = metrics.MicroF1()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     metric = metric.update(y_t, y_p)

            >>> metric
            MicroF1: 0.6

    """

    def __init__(self):
        super().__init__(beta=1.)


class RollingF1(RollingFBeta):
    """Rolling binary F1 score.

    The F1 score is a weighted harmonic mean between precision and recall.

    Parameters:
        window_size (int): Size of the rolling window.

    Attributes:
        precision (metrics.RollingPrecision): Rolling precision score.
        recall (metrics.RollingRecall): Rolling recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [False, False, False, True, True, True]
            >>> y_pred = [False, False, True, True, False, False]

            >>> metric = metrics.RollingF1(window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            0.0
            0.0
            0.0
            0.666666...
            0.5
            0.5

    """

    def __init__(self, window_size):
        super().__init__(beta=1., window_size=window_size)


class RollingMacroF1(RollingMacroFBeta):
    """Rolling macro-average F1 score.

    This works by computing the F1 score per class, and then performs a global average.

    Parameters:
        beta (float): Weight of precision in harmonic mean.
        window_size (int): Size of the rolling window.

    Attributes:
        fbetas (collections.defaultdict): Rolling F-Beta scores per class.
        classes (set): Observed classes.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMacroF1(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMacroF1: 1.
            RollingMacroF1: 0.333333
            RollingMacroF1: 0.555556
            RollingMacroF1: 0.333333
            RollingMacroF1: 0.4

    """

    def __init__(self, window_size):
        super().__init__(beta=1., window_size=window_size)


class RollingMicroF1(RollingMicroFBeta):
    """Rolling micro multiclass F1 score.

    Parameters:
        window_size (int): Size of the rolling window.

    Attributes:
        beta2 (float): Beta squared value.
        precision (metric.RollingPrecision): Rolling precision score.
        recall (metric.RollingRecall): Rolling recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 0]
            >>> y_pred = [0, 1, 1, 2, 1]

            >>> metric = metrics.RollingMicroF1(window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            1.0
            1.0
            0.666666...
            0.666666...
            0.333333...

    """

    def __init__(self, window_size):
        super().__init__(beta=1., window_size=window_size)
