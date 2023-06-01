from methods.base import ChangePointDetector
import numpy as np
import scipy.stats as ss
from itertools import islice
from numpy.linalg import inv
from functools import partial


class MultivariateT:
    def __init__(
        self,
        dims: int = 1,
        dof: int = 0,
        kappa: int = 1,
        mu: float = -1,
        scale: float = -1,
    ):
        """
        Generate a new predictor using the multivariate student T distribution as the posterior predictive.
        This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
        and a Gaussian prior on the mean.
        Implementation based on Haines, T.S., Gaussian Conjugate Prior Cheat Sheet.

        Args:
            dof: The degrees of freedom on the prior distribution of the precision (inverse covariance).
            kappa: The number of observations we've already seen.
            mu: The mean of the prior distribution on the mean.
            scale: The mean of the prior distribution on the precision.
            dims: The number of variables.
        """
        # We default to the minimum possible degrees of freedom, which is 1 greater than the dimensionality
        if dof == 0:
            dof = dims + 1
        if mu == -1:
            mu = [0] * dims # default mean
        else:
            mu = [mu] * dims

        # The default covariance is the identity matrix, so the scale is also the inverse of the identity.
        if scale == -1:
            scale = np.identity(dims)
        else:
            scale = np.identity(scale)

        # Track time
        self.t = 0

        # number of variables 
        self.dims = dims

        # Each parameter is a vector of size 1 x t, where t is time. Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def pdf(self, data: np.array):
        """
        Returns the probability of the observed data under the current and historical parameters.

        Args:
            data: The datapoints to be evaluated (shape: 1 x D vector).

        Raises:
            Exception: If scipy version is less than 1.6.0.
        """

        self.t += 1
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims(
            (self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        ret = np.empty(self.t)

        try:
            # This can't be vectorised due to https://github.com/scipy/scipy/issues/13450
            for i, (df, loc, shape) in islice(
                enumerate(zip(t_dof, self.mu, inv(
                    expanded * self.scale))), self.t
            ):
                ret[i] = ss.multivariate_t.pdf(
                    x=data, df=df, loc=loc, shape=shape)
        except AttributeError:
            raise Exception(
                "You need scipy 1.6.0 or greater to use the multivariate t distribution"
            )
        
        return ret

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a Bayesian update on the prior parameters, given data.

        Args:
            data: The datapoints to be evaluated (shape: 1 x D vector).
        """

        centered = data - self.mu

        # We simultaneously update each parameter in the vector, because following figure 1c of the BOCD paper, each
        # parameter for a given t, r is derived from the same parameter for t-1, r-1
        # Then, we add the prior back in as the first element
        self.scale = np.concatenate(
            [
                self.scale[:1],
                inv(
                    inv(self.scale)
                    + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2))
                    * (np.expand_dims(centered, 2) @ np.expand_dims(centered, 1))
                ),
            ]
        )
        self.mu = np.concatenate(
            [
                self.mu[:1],
                (np.expand_dims(self.kappa, 1) * self.mu + data)
                / np.expand_dims(self.kappa + 1, 1),
            ]
        )
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])


class StudentT():
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):
        """
        StudentT distribution except normal distribution is replaced with the student T distribution.
        https://en.wikipedia.org/wiki/Normal-gamma_distribution

        Args:
            alpha: Alpha in gamma distribution prior.
            beta: Beta in gamma distribution prior.
            mu: Mean from normal distribution.
            kappa: Variance from normal distribution.
        
        Raises:
            ValueError: If any of the parameters are invalid.
        """

        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data: np.array):
        """
        Return the pdf function of the t distribution.

        Args:
            data: The datapoints to be evaluated (shape: 1 x D vector).
        """
        return ss.t.pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) /
                          (self.alpha * self.kappa)),
        )

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a Bayesian update on the prior parameters, given data.

        Args:
            data: The datapoints to be evaluated (shape: 1 x D vector).
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) /
                (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


class BOCPD(ChangePointDetector):
    """
    Bayesian Online Change Point Detection (BOCPD) algorithm.

    This algorithm detects change points in data using a Bayesian framework.

    Args:
        threshold: The threshold value for detecting change points.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Attributes:
        hazard_function: The hazard function used for evaluating the growth probabilities.
        log_likelihood_class: The class representing the log-likelihood function.
        maxes: Array to store the indices with the maximum probabilities.
        R: Matrix for storing the run length probabilities.
        threshold: The threshold value for detecting change points.

    Methods:
        update: Update the BOCPD model with new data.
        constant_hazard: Compute the hazard function for Bayesian online learning.
        _reset: Reset the state of the BOCPD model.
        is_multivariate: Check if the model is multivariate.

    Inherits:
        ChangePointDetector
    """

    def __init__(self, threshold, **kwargs):
        """
        Initialize the BOCPD model.

        Args:
            threshold: The threshold value for detecting change points.
            **kwargs: Additional keyword arguments to be passed to the base class.
        """
        super().__init__(**kwargs)
        self.hazard_function = partial(self.constant_hazard, 20)
        self.log_likelihood_class = StudentT(0.1, .01, 1, 0)

        len_data_estimate = threshold * 10000
        self.maxes = np.zeros(len_data_estimate) # 
        self.R = np.zeros((len_data_estimate, len_data_estimate))
        self.R[0, 0] = 1
        self.threshold = threshold

    def update(self, x, t) -> "ChangePointDetector":
        """
        Update the BOCPD model with new data.

        Args:
            x: The data point to be evaluated.
            t: The time index of the data point.

        Returns:
            ChangePointDetector: The updated ChangePointDetector object.
        """
        # Shift time to start at 0
        t = t-1 
        self._change_point_detected = False

        # Compute the predictive probabilities of the data x
        predprobs = self.log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        H = self.hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities 
        # Shift the probabilities down and to the right, scaled by the hazard function and the predictive probabilities.
        self.R[1: t + 2, t + 1] = self.R[0: t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're accumulating the mass back down at r = 0.
        self.R[0, t + 1] = np.sum(self.R[0: t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical stability.
        self.R[:, t + 1] = self.R[:, t + 1] / np.sum(self.R[:, t + 1])

        # Update the parameter sets for each possible run length.
        self.log_likelihood_class.update_theta(x, t=t)

        # Store the index with the maximum probability
        self.maxes[t] = self.R[:, t].argmax()

        # Check if a change point has been detected
        if self.maxes[t] - self.maxes[t-1] > self.threshold:
            self._change_point_detected = True

        # Return the updated ChangePointDetector object
        return self

    def constant_hazard(self, lam, r):
        """
        Compute the hazard function for Bayesian online learning.

        Args:
            lam: The initial probability.
            r: The R matrix.

        Returns:
            ndarray: The hazard function values.
        """

        return 1 / lam * np.ones(r.shape)

    def _reset(self):
        """
        Reset the state of the BOCPD model.
        """
        super()._reset()
        self.maxes = []
        self.R = []

    def is_multivariate(self):
        """
        Check if the BOCPD model is multivariate.

        Returns:
            bool: True if the model is multivariate, False otherwise.
        """
        return False
