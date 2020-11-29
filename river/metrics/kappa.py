from . import base


__all__ = ["CohenKappa", "KappaT", "KappaM"]


class CohenKappa(base.MultiClassMetric):
    r"""Cohen's Kappa score.

    Cohen's Kappa expresses the level of agreement between two annotators on a classification
    problem. It is defined as

    $$
    \kappa = (p_o - p_e) / (1 - p_e)
    $$

    where $p_o$ is the empirical probability of agreement on the label
    assigned to any sample (prequential accuracy), and $p_e$ is
    the expected agreement when both annotators assign labels randomly.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

    >>> metric = metrics.CohenKappa()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    CohenKappa: 0.428571

    References
    ----------
    [^1]: J. Cohen (1960). "A coefficient of agreement for nominal scales". Educational and Psychological Measurement 20(1):37-46. doi:10.1177/001316446002000104.
    """

    def get(self):

        try:
            p0 = self.cm.sum_diag / self.cm.n_samples  # same as accuracy
        except ZeroDivisionError:
            p0 = 0

        pe = 0

        for c in self.cm.classes:
            estimation_row = self.cm.sum_row[c] / self.cm.n_samples
            estimation_col = self.cm.sum_col[c] / self.cm.n_samples
            pe += estimation_row * estimation_col

        try:
            return (p0 - pe) / (1 - pe)
        except ZeroDivisionError:
            return 0.0


class KappaM(base.MultiClassMetric):
    r"""Kappa-M score.

    The Kappa-M statistic compares performance with the majority class classifier.
    It is defined as

    $$
    \kappa_{m} = (p_o - p_e) / (1 - p_e)
    $$

    where $p_o$ is the empirical probability of agreement on the label
    assigned to any sample (prequential accuracy), and $p_e$ is
    the prequential accuracy of the `majority classifier`.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird', 'cat', 'ant', 'cat', 'cat', 'ant']
    >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat', 'ant', 'ant', 'cat', 'cat', 'ant']

    >>> metric = metrics.KappaM()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    KappaM: 0.25

    References
    ----------
    [1^]: A. Bifet et al. "Efficient online evaluation of big data stream classifiers."
        In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery
        and data mining, pp. 59-68. ACM, 2015.

    """

    def get(self):

        try:
            p0 = self.cm.sum_diag / self.cm.n_samples  # same as accuracy
        except ZeroDivisionError:
            p0 = 0

        try:
            pe = self.cm.weight_majority_classifier / self.cm.n_samples
            return (p0 - pe) / (1.0 - pe)
        except ZeroDivisionError:
            return 0.0


class KappaT(base.MultiClassMetric):
    r"""Kappa-T score.

    The Kappa-T measures the temporal correlation between samples.
    It is defined as

    $$
    \kappa_{t} = (p_o - p_e) / (1 - p_e)
    $$

    where $p_o$ is the empirical probability of agreement on the label
    assigned to any sample (prequential accuracy), and $p_e$ is
    the prequential accuracy of the `no-change classifier` that predicts
    only using the last class seen by the classifier.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

    >>> metric = metrics.KappaT()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    KappaT: 0.6

    References
    ----------
    [^1]: A. Bifet et al. (2013). "Pitfalls in benchmarking data stream classification
        and how to avoid them." Proc. of the European Conference on Machine Learning
        and Principles and Practice of Knowledge Discovery in Databases (ECMLPKDD'13),
        Springer LNAI 8188, p. 465-479.

    """

    def get(self):

        try:
            p0 = self.cm.sum_diag / self.cm.n_samples  # same as accuracy
        except ZeroDivisionError:
            p0 = 0

        try:
            pe = self.cm.weight_no_change_classifier / self.cm.n_samples
            return (p0 - pe) / (1.0 - pe)
        except ZeroDivisionError:
            return 0.0
