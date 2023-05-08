from __future__ import annotations

from river import base, stats


class EDDM(base.BinaryDriftAndWarningDetector):
    r"""Early Drift Detection Method.

    EDDM (Early Drift Detection Method) aims to improve the detection rate of gradual
    concept drift in DDM, while keeping a good performance against abrupt concept drift.

    This method works by keeping track of the average distance between two errors instead
    of only the error rate. For this, it is necessary to keep track of the running average
    distance and the running standard deviation, as well as the maximum distance and the maximum
    standard deviation.

    The algorithm works similarly to the DDM algorithm, by keeping track of statistics only. It works
    with the running average distance ($p_i'$) and the running standard deviation ($s_i'$), as
    well as $p'_{max}$ and $s'_{max}$, which are the values of $p_i'$ and $s_i'$ when
    $(p_i' + 2 * s_i')$ reaches its maximum.

    Like DDM, there are two threshold values that define the borderline between no change, warning zone,
    and drift detected. These are as follows:

    * if $(p_i' + 2 * s_i') / (p'_{max} + 2 * s'_{max}) < \alpha$ -> Warning zone

    * if $(p_i' + 2 * s_i') / (p'_{max} + 2 * s'_{max}) < \beta$ -> Change detected

    $\alpha$ and $\beta$ are set to 0.95 and 0.9, respectively.

    **Input:** `x` is an entry in a stream of bits, where 1 indicates error/failure and 0
    represents correct/normal values.

    For example, if a classifier's prediction $y'$ is right or wrong w.r.t. the
    true target label $y$:

    - 0: Correct, $y=y'$

    - 1: Error, $y \\neq y'$

    Parameters
    ----------
    warm_start
         The minimum required number of monitored errors/failures so change can be detected.
        Warm start parameter for the drift detector.
    alpha
        Threshold for triggering a warning. Must be between 0 and 1. The smaller the value, the
        more conservative the detector becomes.
    beta
        Threshold for triggering a drift. Must be between 0 and 1. The smaller the value, the
        more conservative the detector becomes.

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> # Change the default hyperparameters to avoid too many false positives
    >>> # in this example
    >>> eddm = drift.binary.EDDM(alpha=0.8, beta=0.75)

    >>> # Simulate a data stream where the first 1000 instances come from a uniform distribution
    >>> # of 1's and 0's
    >>> data_stream = rng.choices([0, 1], k=1000)
    >>> # Increase the probability of 1's appearing in the next 1000 instances
    >>> data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])

    >>> print_warning = True
    >>> # Update drift detector and verify if change is detected
    >>> for i, x in enumerate(data_stream):
    ...     _ = eddm.update(x)
    ...     if eddm.warning_detected and print_warning:
    ...         print(f"Warning detected at index {i}")
    ...         print_warning = False
    ...     if eddm.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         print_warning = True
    Warning detected at index 1059
    Change detected at index 1278

    References
    ----------
    [^1]: Early Drift Detection Method. Manuel Baena-Garcia, Jose Del Campo-Avila, Raúl Fidalgo, Albert Bifet, Ricard Gavalda, Rafael Morales-Bueno. In Fourth International Workshop on Knowledge Discovery from Data Streams, 2006.

    """

    def __init__(
        self,
        warm_start: int = 30,
        alpha: float = 0.95,
        beta: float = 0.9,
    ):
        super().__init__()
        self.warm_start = warm_start

        if alpha < beta:
            raise ValueError("'alpha' must be greater or equal to 'beta'.")

        self.alpha = alpha
        self.beta = beta

        self._reset()

    def _reset(self):
        super()._reset()

        # Variance of the distance between two error/failure reports
        self._error_distances = stats.Var()
        # Number of observations
        self._n = 0
        # Index of the last observed error
        self._last_error = 0
        # Number of observed errors/failures
        self._n_errors = 0

        self._p2s_prime_max = -1

    def update(self, x):
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            This parameter indicates whether the last sample analyzed was correctly classified or
            not. 1 indicates an error (miss-classification).

        Returns
        -------
        self

        """

        if self.drift_detected:
            self._reset()

        # Update the sample counter
        self._n += 1

        # Error/failure
        if x == 1:
            self._n_errors += 1
            # Monitor the interval between errors/failures
            self._error_distances.update(self._n - self._last_error)

            if self._n > self.warm_start:
                # Mean and variance of the intervals between errors
                pi_prime = self._error_distances.mean.get()
                si_prime = self._error_distances.get() ** 0.5

                p2s_prime = pi_prime + 2 * si_prime
                if p2s_prime > self._p2s_prime_max:
                    self._p2s_prime_max = p2s_prime
                elif self._n_errors > self.warm_start:
                    level = p2s_prime / self._p2s_prime_max
                    if level < self.beta:
                        self._drift_detected = True
                    elif level < self.alpha:
                        self._warning_detected = True
                    else:
                        self._warning_detected = False

            # Update the index of the last error/failure detected
            self._last_error = self._n

        return self
