"""Online Principal Component Analysis (PCA) in [River API](riverml.xyz).

This module contains the implementation of the Online PCA algorithm.
It is based on the paper by Eftekhari et al. [^1]

References:
    [^1]: Eftekhari, A., Ongie, G., Balzano, L., Wakin, M. B. (2019). Streaming Principal Component Analysis From Incomplete Data. Journal of Machine Learning Research, 20(86), pp.1-62. url:http://jmlr.org/papers/v20/16-627.html.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from typing import Any

import numpy as np

from river.base import Transformer

__all__ = [
    "OnlinePCA",
]


class OnlinePCA(Transformer):
    """Online Principal Component Analysis (PCA).

    Args:
        n_components: Desired dimensionality of output data. The default value is useful for visualisation.
        b: size of the blocks. Must be greater than or equal to n_components.
        lambda_: tuning parameter
        sigma: reject threshold
        tau: reject threshold

    Attributes:
        feature_names_in_: List of input features.
        n_seen: Number of samples seen.
        Y_k: Block of received data of size (n_features_in_, b).
        S_hat: R-dimensional subspace with orthonormal basis (n_features_in_, n_components)

    Examples:
        >>> import pandas as pd
        >>> np.random.seed(0)
        >>> m = 20
        >>> n = 80
        >>> mean = [5, 10, 15]
        >>> covariance_matrix = [[1, 0.5, 0.3],
        ...                      [0.5, 1, 0.2],
        ...                      [0.3, 0.2, 1]]
        >>> num_samples = 100
        >>> X = np.random.multivariate_normal(mean, covariance_matrix, num_samples)
        >>> n_nans = 2
        >>> nan_indices = np.random.choice(range(X.shape[0]), size=n_nans, replace=False)
        >>> X[nan_indices] = np.nan
        >>> X = pd.DataFrame(X)
        >>> pca = OnlinePCA(n_components=2)
        >>> for _, x in X.iloc[:50].iterrows():
        ...     pca.learn_one(x.to_dict())
        >>> pca.transform_one(X.iloc[-1, :].to_dict())  # doctest: +SKIP
        {0: -17.8587, 1: -1.5643}

        >>> pca = OnlinePCA(n_components=2, b=4)
        >>> for _, x in X.iloc[:50].iterrows():
        ...     pca.learn_one(x.to_dict())
        >>> pca.transform_one(X.iloc[-1, :].to_dict())  # doctest: +SKIP
        {0: -17.9470, 1: -1.0941}

    """

    def __init__(
        self,
        n_components: int = 2,
        b: int | None = None,
        lambda_: float = 0.0,
        sigma: float = 0.0,
        tau: float = 0.0,
        seed: int | None = None,
    ):
        """Initialize the OnlinePCA model."""
        self.n_components = int(n_components)
        # Default maximizes the efficiency [Eftekhari, et al. (2019)]
        if not b:
            b = self.n_components
        else:
            b = int(b)
        self.b = b
        if lambda_ < 0:
            raise ValueError("lambda_ must be >= 0")
        self.lambda_ = lambda_
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = sigma
        if tau < 0:
            raise ValueError("tau must be >= 0")
        self.tau = tau

        self.feature_names_in_: list[str]
        self.n_features_in_: int  # n [Eftekhari, et al. (2019)]
        self.n_seen: int = 0  # k [Eftekhari, et al. (2019)]
        self.Y_k: deque[np.ndarray]
        self.P_omega_k: deque[np.ndarray]
        self.S_hat: np.ndarray
        self.seed = seed
        np.random.seed(self.seed)

    def learn_one(self, x: dict[Hashable, Any]) -> None:
        """Learn one sample from the data.

        Args:
            x: Incomplete observation of data matrix. Accepts NaNs (n_features_in_,)
        """
        if self.n_seen == 0:
            self.feature_names_in_ = [str(k) for k in x.keys()]
        else:
            if set(self.feature_names_in_).difference(str(k) for k in x.keys()):
                raise ValueError("Input features do not match the features seen during training.")
        x_arr = np.array(list(x.values()))
        if self.n_seen == 0:
            self.n_features_in_ = x_arr.shape[0]
            if self.n_components == 0:
                self.n_components = self.n_features_in_
            # Make b feasible if not set and learn_one is called first
            if not self.b:
                self.b = self.n_components
            self.Y_k = deque(maxlen=self.b)
            self.P_omega_k = deque(maxlen=self.b)
            # Initialize S_hat with random orthonormal matrix for transform_one
            r_mat = np.random.randn(self.n_features_in_, self.n_components)
            self.S_hat, _ = np.linalg.qr(r_mat)

        # Random index set over which s_t is observed
        omega_t = ~np.isnan(x_arr)  # (n_features_in_,)
        x_arr = np.nan_to_num(x_arr, nan=0.0)
        # Projection onto coordinate set. Diagonal entry corresponding to the index set omega_t (n_features_in_, n_features_in_)
        P_omega_t = np.diag(omega_t).astype(int)
        self.Y_k.append(x_arr)
        self.P_omega_k.append(P_omega_t)

        if len(self.Y_k) == self.b:
            # Reinitialize S_hat now when deque is full
            if self.n_seen == self.b - 1:
                # Let S_hat \in \mathbb{R}^{n \times b} be the
                _, _, V = np.linalg.svd(np.array(self.Y_k), full_matrices=False)
                self.S_hat = V.T[:, : self.n_components]
            else:
                R_k = np.empty((self.n_features_in_, self.b))
                # range((self.n_seen - 1) * self.b + 1, self.n_seen * self.b) [Eftekhari, et al. (2019)]
                for k, (y_t, P_omega_t) in enumerate(zip(self.Y_k, self.P_omega_k)):
                    P_omega_t_comp = np.identity(self.n_features_in_) - P_omega_t

                    I_r = np.identity(self.n_components)
                    S_hat_t = self.S_hat.T
                    R_k[:, k] = (
                        y_t
                        + P_omega_t_comp
                        @ self.S_hat
                        @ np.linalg.pinv(S_hat_t @ P_omega_t @ self.S_hat + self.lambda_ * I_r)
                        @ S_hat_t
                        @ y_t
                    )
                U_r, sigma_r, _ = np.linalg.svd(R_k)
                _sigma_below_thresh = sigma_r[self.n_components - 1] < self.sigma
                if self.b > self.n_components:
                    _sigma_ratio_below_thresh = (
                        sigma_r[self.n_components] <= (1 + self.tau) * sigma_r[1]
                    )
                else:
                    _sigma_ratio_below_thresh = True
                if not (_sigma_below_thresh or _sigma_ratio_below_thresh):
                    self.S_hat = U_r[:, : self.n_components]

            self.Y_k.clear()  # Non overlapping blocks

        self.n_seen += 1

    def transform_one(self, x: dict[Hashable, Any]) -> dict[Hashable, Any]:
        """Transform one sample from the data.

        Args:
            x: Incomplete observation of data matrix. Accepts NaNs (n_features_in_,)
        """
        x_arr = np.array(list(x.values()))
        # If transform one is called before any learning has been done
        if not hasattr(self, "S_hat"):
            return dict(
                zip(
                    range(self.n_components),
                    np.zeros(self.n_components),
                )
            )
        x_arr = x_arr @ self.S_hat
        return dict(zip(range(self.n_components), x_arr))
