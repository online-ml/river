# -*- coding: utf-8 -*-
"""Dynamic Mode Decomposition (DMD) in scikkit-learn API.

This module contains the implementation of the Online DMD, Windowed DMD,
and DMD with Control algorithm. It is based on the paper by Zhang et al.
[^1] and implementation of authors available at [GitHub](https://github.com/haozhg/odmd).
However, this implementation provides a more flexible interface aligned with
River API covers and separates update and revert methods in Windowed DMD.

References:
    [^1]: Schmid, P. (2022). Dynamic Mode Decomposition and Its Variants. 54(1), pp.225-254. doi:[10.1146/annurev-fluid-030121-015835](https://doi.org/10.1146/annurev-fluid-030121-015835).
"""
import numpy as np


class DMD:
    """Class for Dynamic Mode Decomposition (DMD) model.

    Args:
        r: Number of modes to keep. If 0 (default), all modes are kept.

    Attributes:
        m: Number of variables.
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
        self._C: np.ndarray | None
        self._xi: np.ndarray
        self._Y: np.ndarray

    @property
    def C(self) -> np.ndarray:
        if self._C is None:
            self._C = np.vander(self.Lambda, self.n, increasing=True)
        return self._C

    @property
    def xi(self) -> np.ndarray:
        if not hasattr(self, "_xi") or self._xi is None:
            # self._xi = self.Phi.conj().T @ self._Y @ np.linalg.pinv(self.C)
            import cvxpy as cp

            gamma = 0.5
            xi = cp.Variable(self.m)
            objective = cp.Minimize(
                cp.norm(self._Y - self.Phi @ cp.diag(xi) @ self.C, "fro")
                + gamma * cp.norm(xi, 1)
            )
            # As Quadratic Programming
            # FIX: ValueError: The 'minimize' objective must be real valued.
            # XHX = np.dot(X.T, X)
            # CCH = np.dot(C, C.T)
            # P = np.multiply(XHX, CCH.conjugate())
            # p_star = np.dot(C, np.dot(v.T, np.dot(sigma, X)))
            # print(sigma)
            # print()
            # s = sigma.T @ sigma

            # # Extract the optimal value
            # objective = cp.Minimize(xi.flatten().T @ P @ xi.flatten() - p_star.T @ xi.flatten() + s)
            problem = cp.Problem(objective)

            # Solve the problem
            problem.solve()
            self._xi = xi.value
        return self._xi

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        # Perform singular value decomposition on X
        u_, sigma, v = np.linalg.svd(X, full_matrices=False)
        r = self.r if self.r > 0 else len(sigma)
        sigma_inv = np.reciprocal(sigma[:r])
        # Compute the low-rank approximation of Koopman matrix
        self.A_bar = (
            u_[: self.m, :r].conj().T
            @ Y
            @ v[:r, :].conj().T
            @ np.diag(sigma_inv)
        )

        # Perform eigenvalue decomposition on A
        self.Lambda, W = np.linalg.eig(self.A_bar)

        # Compute the coefficient matrix
        # TODO: Find out whether to use X or Y (X usage ~ u @ W obviously)
        # self.Phi = X @ v[: r, :].conj().T @ np.diag(sigma_inv) @ W
        self.Phi = u_[:, :r] @ W
        # self.A = self.Phi @ np.diag(self.Lambda) @ np.linalg.pinv(self.Phi)
        self.A = (
            Y @ v[:r, :].conj().T @ np.diag(sigma_inv) @ u_[:, :r].conj().T
        )

    def fit(self, X: np.ndarray):
        """
        Fit the DMD model to the input X.

        Args:
            X: Input X matrix of shape (m, n), where m is the number of variables and n is the number of time steps.

        """
        # Build X matrices
        X = X[:, :-1]
        if hasattr(self, "m"):
            self._Y = X[: self.m, 1:]
        else:
            self._Y = X[:, 1:]

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

        mat = np.zeros((self.m, forecast + 1))
        mat[:, 0] = x
        for s in range(1, forecast + 1):
            mat[:, s] = (self.A @ mat[:, s - 1]).real
        return mat[:, 1:]


class DMDwC(DMD):
    def __init__(self, r: int):
        super().__init__(r)
        self.B: np.ndarray
        self.l: int

    def fit(self, X: np.ndarray, u: np.ndarray, B: np.ndarray | None = None):
        # Need to copy u because it will be modified
        F = u.copy()

        self.l = F.shape[0]
        self.m, self.n = X.shape
        if X.shape[1] != F.shape[1]:
            raise ValueError(
                "X and u must have the same number of time steps.\n"
                f"X: {X.shape[1]}, u: {F.shape[1]}"
            )
        if B is None:
            X = np.vstack((X, F))

            X = X[:, :-1]
            self._Y = X[: self.m, 1:]
        else:
            X = X[:, :-1]
            self._Y = X[:, 1:] - B * F[:, :-1]
        # self.m, self.n = self._Y.shape

        super()._fit(X, self._Y)
        # split K into state transition matrix and control matrix
        self.B = self.A[:, -self.l :]
        self.A = self.A[:, : -self.l]

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
        if forecast != 1 and u.shape[1] != forecast:
            raise ValueError(
                "u must have forecast number of time steps.\n"
                f"u: {u.shape[1]}, forecast: {forecast}"
            )

        mat = np.zeros((self.m, forecast + 1))
        mat[:, 0] = x
        for s in range(1, forecast + 1):
            action = (self.B @ u[:, s - 1]).real
            mat[:, s] = (self.A @ mat[:, s - 1]).real + action
        return mat[:, 1:]
