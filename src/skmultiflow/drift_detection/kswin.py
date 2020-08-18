import numpy as np
from scipy import stats
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class KSWIN(BaseDriftDetector):
    r""" Kolmogorov-Smirnov Windowing method for concept drift detection.

    Parameters
    ----------
    alpha: float (default=0.005)
        Probability for the test statistic of the Kolmogorov-Smirnov-Test
        The alpha parameter is very sensitive, therefore should be set
        below 0.01.

    window_size: float (default=100)
        Size of the sliding window

    stat_size: float (default=30)
        Size of the statistic window

    data: numpy.ndarray of shape (n_samples, 1) (default=None,optional)
        Already collected data to avoid cold start.

    Notes
    -----
    KSWIN (Kolmogorov-Smirnov Windowing) [1]_ is a concept change detection method based
    on the Kolmogorov-Smirnov (KS) statistical test. KS-test is a statistical test with
    no assumption of underlying data distribution. KSWIN can monitor data or performance
    distributions. Note that the detector accepts one dimensional input as array.

    KSWIN maintains a sliding window :math:`\Psi` of fixed size :math:`n` (window_size). The
    last :math:`r` (stat_size) samples of :math:`\Psi` are assumed to represent the last
    concept considered as :math:`R`. From the first :math:`n-r` samples of :math:`\Psi`,
    :math:`r` samples are uniformly drawn, representing an approximated last concept :math:`W`.

    The KS-test is performed on the windows :math:`R` and :math:`W` of the same size. KS
    -test compares the distance of the empirical cumulative data distribution :math:`dist(R,W)`.

    A concept drift is detected by KSWIN if:

    * :math:`dist(R,W) > \sqrt{-\frac{ln\alpha}{r}}`

    -> The difference in empirical data distributions between the windows :math:`R` and :math:`W`
    is too large as that R and W come from the same distribution.

    References
    ----------
    .. [1] Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive
       Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> from skmultiflow.drift_detection import KSWIN
    >>> import numpy as np
    >>> # Initialize KSWIN and a data stream
    >>> kswin = KSWIN(alpha=0.01)
    >>> stream = SEAGenerator(classification_function = 2,
    >>>     random_state = 112, balance_classes = False,noise_percentage = 0.28)
    >>> # Store detections
    >>> detections = []
    >>> # Process stream via KSWIN and print detections
    >>> for i in range(1000):
    >>>         data = stream.next_sample(10)
    >>>         batch = data[0][0][0]
    >>>         kswin.add_element(batch)
    >>>         if kswin.detected_change():
    >>>             print("\rIteration {}".format(i))
    >>>             print("\r KSWINReject Null Hyptheses")
    >>>             detections.append(i)
    >>> print("Number of detections: "+str(len(detections)))
    """

    def __init__(self, alpha=0.005, window_size=100, stat_size=30, data=None):
        super().__init__()
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.change_detected = False
        self.p_value = 0
        self.n = 0
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.window_size < 0:
            raise ValueError("window_size must be greater than 0")

        if self.window_size < self.stat_size:
            raise ValueError("stat_size must be smaller than window_size")

        if not isinstance(data, np.ndarray) or data is None:
            self.window = np.array([])
        else:
            self.window = data

    def add_element(self, input_value):
        """ Add element to sliding window

        Adds an element on top of the sliding window and removes
        the oldest one from the window. Afterwards, the KS-test
        is performed.

        Parameters
        ----------
        input_value: ndarray
            New data sample the sliding window should add.
        """
        self.n += 1
        currentLength = self.window.shape[0]
        if currentLength >= self.window_size:
            self.window = np.delete(self.window, 0)
            rnd_window = np.random.choice(self.window[:-self.stat_size], self.stat_size)

            (st, self.p_value) = stats.ks_2samp(rnd_window,
                                                self.window[-self.stat_size:], mode="exact")

            if self.p_value <= self.alpha and st > 0.1:
                self.change_detected = True
                self.window = self.window[-self.stat_size:]
            else:
                self.change_detected = False
        else:  # Not enough samples in sliding window for a valid test
            self.change_detected = False

        self.window = np.concatenate([self.window, [input_value]])

    def detected_change(self):
        """ Get detected change

        Returns
        -------
        bool
            Whether or not a drift occurred

        """
        return self.change_detected

    def reset(self):
        """ reset

        Resets the change detector parameters.
        """
        self.p_value = 0
        self.window = np.array([])
        self.change_detected = False
