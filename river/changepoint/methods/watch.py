from methods.base import ChangePointDetector
from scipy.stats import wasserstein_distance


class WATCH(ChangePointDetector):
    """
    WATCH (Wasserstein Change Point Detection for High-Dimensional Time Series Data) is a change point detector
    that uses the Wasserstein distance to detect changes in the underlying distribution of a high-dimensional
    time series data stream. It maintains a distribution of samples and compares the Wasserstein distance between
    new batches of samples and the distribution.

    Args:
        batch_size: Size of each batch of samples.
        min_distr_size: Minimum size of the distribution before change point detection starts.
        max_distr_size: Maximum size of the distribution. When the size exceeds this value, old samples are removed.
        epsilon: Scaling factor for the maximum Wasserstein distance used for change point detection.
        **kwargs: Additional keyword arguments.

    Methods:
        compute_eta(): Compute the scaling factor for the maximum Wasserstein distance.
        update(x, t): Update the change point detector with a new data point.
        _reset(): Reset the change point detector to its initial state.
        is_multivariate(): Check if the change point detector is designed for multivariate data.

    """
    def __init__(self, batch_size, min_distr_size, max_distr_size, epsilon, **kwargs):
        """
        Initialize the WATCH change point detector.

        Args:
            batch_size: Size of each batch of samples.
            min_distr_size: Minimum size of the distribution before change point detection starts.
            max_distr_size: Maximum size of the distribution. When the size exceeds this value, old samples are removed.
            epsilon: Scaling factor for the maximum Wasserstein distance used for change point detection.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(**kwargs)
        self.distr = []
        self.current_batch = []
        self.R = []
        self.eta = 0
        self.batch_size = batch_size
        self.max_distr_size = max_distr_size  # mu
        self.min_distr_size = min_distr_size  # kappa
        self.epsilon = epsilon

    def compute_eta(self):
        """
        Compute the scaling factor for the maximum Wasserstein distance.

        """
        batches_in_distr = len(self.distr)//self.batch_size
        max_wasserstein_distance = 0
        for i in range(batches_in_distr):
            batch = self.distr[i*self.batch_size:(i+1)*self.batch_size]
            assert (len(batch) == self.batch_size)
            if wasserstein_distance(batch, self.distr) > max_wasserstein_distance:
                max_wasserstein_distance = wasserstein_distance(
                    batch, self.distr)
        self.eta = self.epsilon * max_wasserstein_distance

    def update(self, x, t) -> "ChangePointDetector":
        """
        Update the change point detector with a new data point.

        Args:
            x: The new data point.
            t: The time step or index of the new data point.

        Returns:
            ChangePointDetector: The updated change point detector.

        """
        self._change_point_detected = False
        # Adding samples to the batch
        self.current_batch.append(x)
        # When the batch reaches the window size, find if there is a change point
        if len(self.current_batch) == self.batch_size:
            # If the distribution is too small, add the current batch of samples to it
            if len(self.distr) < self.min_distr_size:
                for sample in self.current_batch:
                    self.distr.append(sample)
                # If the distribution becomes big enough, compute the maximum wasserstein distance in the distribution * epsilon
                if len(self.distr) >= self.min_distr_size:
                    self.compute_eta()  # Change
            else:
                # The distribution is big enough and we compute the wasserstein distance
                nu = wasserstein_distance(self.current_batch, self.distr)
                # If the wasserstein distance between the new batch and distribution is bigger than the biggest wasserstein distance among the distribution
                if nu > self.eta:
                    self._change_point_detected = True
                    # Create change point
                    self.R.append(t)
                    # Reset the distribution to the current minibatch only
                    self.distr = self.current_batch
                    self._change_point_score = nu
                else:
                    # If the distribution is not too big
                    if len(self.distr) < self.max_distr_size:
                        # Add the current minibatch to the distribution
                        for sample in self.current_batch:
                            self.distr.append(sample)
                        self.compute_eta()  # Recompute eta

            self.current_batch = []  # Reset the minibatch

        return self

    def _reset(self):
        """
        Reset the change point detector to its initial state.

        """
        super()._reset()
        # self.R = []
        self.distr = []
        self.current_batch = []

    def is_multivariate(self):
        """
        Check if the change point detector is designed for multivariate data.

        Returns:
            bool: True if the change point detector is designed for multivariate data, False otherwise.

        """
        return False
