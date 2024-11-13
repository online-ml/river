import numpy as np


class RLS(object):
    """
    Recursive Least Squares (RLS)

    The Recursive Least Squares (RLS) algorithm is an adaptive filtering method that adjusts filter coefficients
    to minimize the weighted least squares error between the desired and predicted outputs. It is widely used
    in signal processing and control systems for applications requiring fast adaptation to changes in input signals.

    Parameters
    ----------
    p : int
        The order of the filter (number of coefficients to be estimated).
    l : float, optional, default=0.99
        Forgetting factor (0 < l ≤ 1). Controls how quickly the algorithm forgets past data.
        A smaller value makes the algorithm more responsive to recent data.
    delta : float, optional, default=1000000
        Initial value for the inverse correlation matrix (P(0)). A large value ensures numerical stability at the start.

    Attributes
    ----------
    p : int
        Filter order.
    l : float
        Forgetting factor.
    delta : float
        Initialization value for P(0).
    currentStep : int
        The current iteration step of the RLS algorithm.
    x : numpy.ndarray
        Input vector of size (p+1, 1). Stores the most recent inputs.
    P : numpy.ndarray
        Inverse correlation matrix, initialized to a scaled identity matrix.
    estimates : list of numpy.ndarray
        List of estimated weight vectors (filter coefficients) at each step.
    Pks : list of numpy.ndarray
        List of inverse correlation matrices (P) at each step.

    Methods
    -------
    estimate(xn, dn)
        Updates the filter coefficients using the current input (`xn`) and desired output (`dn`).


    Examples
    --------
    >>> import numpy as np

    >>> # Initialize the RLS filter with order 2, forgetting factor 0.98, and delta 1e6
    >>> rls = RLS(p=2, l=0.98, delta=1e6)

    >>> # Simulate some data
    >>> np.random.seed(42)
    >>> num_samples = 100
    >>> x_data = np.sin(np.linspace(0, 10, num_samples))  # Input signal
    >>> noise = np.random.normal(0, 0.1, num_samples)    # Add some noise
    >>> d_data = 0.5 * x_data + 0.3 + noise              # Desired output

    >>> # Apply RLS algorithm
    >>> for xn, dn in zip(x_data, d_data):
    ...     weights = rls.estimate(xn, dn)
    ...     print("Updated Weights:", weights.flatten())

    >>> # Final weights after adaptation
    >>> print("Final Weights:", rls.estimates[-1].flatten())
    """
    def __init__(self, p: int, l=0.99, delta=1000000):
        """
            Initializes the Recursive Least Squares (RLS) filter.

            Parameters
            ----------
            p : int
                Filter order (number of coefficients).
            l : float, optional
                Forgetting factor (default is 0.99).
            delta : float, optional
                Initial value for the inverse correlation matrix (default is 1,000,000).
        """
        self.p = p  # Filter order
        self.l = l  # Forgetting factor
        self.delta = delta  # Value to initialise P(0)

        self.currentStep = 0

        self.x = np.zeros((p + 1, 1))  # Column vector
        self.P = np.identity(p + 1) * self.delta

        self.estimates = []
        self.estimates.append(np.zeros((p + 1, 1)))  # Weight vector initialized to zeros

        self.Pks = []
        self.Pks.append(self.P)

    def estimate(self, xn: float, dn: float):
        """
            Performs one iteration of the RLS algorithm to update filter coefficients.

            Parameters
            ----------
            xn : float
                The current input sample.
            dn : float
                The desired output corresponding to the current input.

            Returns
            -------
            numpy.ndarray
                Updated weight vector (filter coefficients) after the current iteration.
            """
        # Update input vector
        self.x = np.roll(self.x, -1)
        self.x[-1, 0] = xn

        # Get previous weight vector
        wn_prev = self.estimates[-1]

        # Compute gain vector
        denominator = self.l + self.x.T @ self.Pks[-1] @ self.x
        gn = (self.Pks[-1] @ self.x) / denominator

        # Compute a priori error
        alpha = dn - (self.x.T @ wn_prev)

        # Update inverse correlation matrix
        Pn = (self.Pks[-1] - gn @ self.x.T @ self.Pks[-1]) / self.l
        self.Pks.append(Pn)

        # Update weight vector
        wn = wn_prev + gn * alpha
        self.estimates.append(wn)

        self.currentStep += 1

        return wn
