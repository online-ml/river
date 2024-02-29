# -*- coding: utf-8 -*-
"""Dynamic Mode Decomposition (DMD) in scikkit-learn API.

This module contains the implementation of the Online DMD, Windowed DMD,
and DMD with Control algorithm. It is based on the paper by Zhang et al.
[^1] and implementation of authors available at [GitHub](https://github.com/haozhg/odmd).
However, this implementation provides a more flexible interface aligned with
River API covers and separates update and revert methods in Windowed DMD.

TODO:

    - [ ] Align design with (n, m) convention (currently (m, n)).

References:
    [^1]: Schmid, P. (2022). Dynamic Mode Decomposition and Its Variants. 54(1), pp.225-254. doi:[10.1146/annurev-fluid-030121-015835](https://doi.org/10.1146/annurev-fluid-030121-015835).
"""
from typing import Union

import numpy as np
import scipy as sp


class DMD:
    """Class for Dynamic Mode Decomposition (DMD) model.

    Args:
        r: Number of modes to keep. If 0 (default), all modes are kept.

    Attributes:
        m: Number of features (variables).
        n: Number of time steps (snapshots).
        feature_names_in_: list of feature names. Used for pd.DataFrame inputs.
        Lambda: Eigenvalues of the Koopman matrix.
        Phi: Eigenfunctions of the Koopman operator (Modal structures)
        A_bar: Low-rank approximation of the Koopman operator (Rayleigh quotient matrix).
        A: Koopman operator.
        C: Discrete temporal dynamics matrix (Vandermonde matrix).
        xi: Amlitudes of the singular values of the input matrix.
        _Y: Data snaphot from time step 2 to n (for xi comp.).

    References:
        [^1]: Schmid, P. (2022). Dynamic Mode Decomposition and Its Variants. 54(1), pp.225-254. doi:[10.1146/annurev-fluid-030121-015835](https://doi.org/10.1146/annurev-fluid-030121-015835).
    """

    def __init__(self, r: int = 0):
        self.r = r
        self.m: int
        self.n: int
        self.feature_names_in_: list[str]
        self.Lambda: np.ndarray
        self.Phi: np.ndarray
        self.A_bar: np.ndarray
        self.A: np.ndarray
        self._Y: np.ndarray

    @property
    def C(self) -> np.ndarray:
        return np.vander(self.Lambda, self.n, increasing=True)

    @property
    def xi(self) -> np.ndarray:
        from scipy.optimize import minimize

        def objective_function(x):
            return np.linalg.norm(
                self._Y - self.Phi @ np.diag(x) @ self.C, "fro"
            ) + 0.5 * np.linalg.norm(x, 1)

        # Minimize the objective function
        xi = minimize(objective_function, np.ones(self.m)).x
        self._xi = xi
        return self.xi

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        # Perform singular value decomposition on X
        r = self.r if self.r > 0 else self.m
        # u_, sigma, v = np.linalg.svd(X, full_matrices=False)
        # # Truncate the singular value matrices
        if r < self.m:
            u_, sigma, v = sp.sparse.linalg.svds(X, k=r)
        else:
            u_, sigma, v = np.linalg.svd(X)
        u_, sigma, v = u_[:, :r], sigma[:r], v[:r, :]
        sigma_inv = np.reciprocal(sigma)
        # Compute the low-rank approximation of Koopman matrix
        self.A_bar = u_.conj().T @ Y @ v.conj().T @ np.diag(sigma_inv)

        # Perform eigenvalue decomposition on A
        self.Lambda, W = np.linalg.eig(self.A_bar)

        # Compute the coefficient matrix
        # TODO: Find out whether to use X or Y (X usage ~ u @ W obviously)
        # self.Phi = X @ v[: r, :].conj().T @ np.diag(sigma_inv) @ W
        self.Phi = u_ @ W
        # self.A = self.Phi @ np.diag(self.Lambda) @ np.linalg.pinv(self.Phi)
        self.A = Y @ v.conj().T @ np.diag(sigma_inv) @ u_.conj().T

    def fit(self, X: np.ndarray, Y: Union[np.ndarray, None] = None):
        """
        Fit the DMD model to the input X.

        Args:
            X: Input X matrix of shape (n, m), where m is the number of variables and n is the number of time steps.
            Y: The output snapshot matrix of shape (n, m).

        """
        # Build X matrices
        if Y is None:
            Y = X[1:, :]
            X = X[:-1, :]
        X = X.T  # PATCH#1: Match (m, n) implementation
        Y = Y.T  # PATCH#1: Match (m, n) implementation

        self._Y = Y

        self.m, self.n = self._Y.shape

        self._fit(X, self._Y)

    def predict(
        self,
        x: np.ndarray,
        forecast: int = 1,
    ) -> np.ndarray:
        """
        Predict future values using the trained DMD model.

        Args:
        x: numpy.ndarray of shape (m,)
        forecast: int
            Number of steps to predict into the future.

        Returns:
            predictions: Predicted data matrix for the specified number of prediction steps.
        """
        if self.A is None or self.m is None:
            raise RuntimeError("Fit the model before making predictions.")

        mat = np.zeros((forecast + 1, self.m))
        mat[0, :] = x
        for s in range(1, forecast + 1):
            mat[s, :] = (self.A @ mat[s - 1, :]).real
        return mat[1:, :]


class DMDwC(DMD):
    def __init__(self, r: int, B: Union[np.ndarray, None] = None):
        super().__init__(r)
        self.B = B
        self.known_B = B is not None
        self.l: int

    def fit(
        self, X: np.ndarray, U: np.ndarray, Y: Union[np.ndarray, None] = None
    ):
        U_ = U.copy()
        if not self.known_B:
            X = np.hstack((X, U_))
        if Y is None:
            Y = X[1:, :]
            X = X[:-1, :]
            U_ = U_[:-1, :]

        if X.shape[0] != U_.shape[0]:
            raise ValueError(
                "X and u must have the same number of time steps.\n"
                f"X: {X.shape[0]}, u: {U_.shape[0]}"
            )

        X = X.T  # PATCH#1: Match (m, n) implementation
        U_ = U_.T  # PATCH#1: Match (m, n) implementation
        Y = Y.T  # PATCH#1: Match (m, n) implementation

        if not self.known_B:
            self._Y = Y
        else:
            # Subtract the effect of actuation
            self._Y = Y - self.B * U_[:, :-1]

        self.l = U_.shape[0]
        self.m, self.n = X.shape

        super()._fit(X, self._Y)
        if not self.known_B:
            # split K into state transition matrix and control matrix
            self.B = self.A[: self.m - self.l, -self.l :]
            self.A = self.A[: self.m - self.l, : -self.l]

    def predict(
        self,
        x: np.ndarray,
        u: np.ndarray,
        forecast: int = 1,
    ) -> np.ndarray:
        """
        Predict future values using the trained DMD model.

        Args:
        - forecast: int
            Number of steps to predict into the future.

        Returns:
        - predictions: numpy.ndarray
            Predicted data matrix for the specified number of prediction steps.
        """
        if self.A is None or self.m is None:
            raise RuntimeError("Fit the model before making predictions.")
        if forecast != 1 and u.shape[0] != forecast:
            raise ValueError(
                "u must have forecast number of time steps.\n"
                f"u: {u.shape[1]}, forecast: {forecast}"
            )

        mat = np.zeros((forecast + 1, self.m - self.l))
        mat[0, :] = x
        for s in range(1, forecast + 1):
            action = (self.B @ u[s - 1, :]).real
            mat[s, :] = (self.A @ mat[s - 1, :]).real + action
        return mat[1:, :]
