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

    Attributes:
        C (numpy.ndarray): Discrete temporal dynamics matrix (Vandermonde matrix).
        Y (numpy.ndarray): Data snaphot from time step 2 to n.
        K (numpy.ndarray): Koopman operator.
        Lambda (numpy.ndarray): Eigenvalues of the Koopman matrix.
        m (int): Number of variables.
        n (int): Number of time steps (snapshots).
        r (int): Number of modes to keep. If 0 (default), all modes are kept.
        Phi (numpy.ndarray): Eigenfunctions of the Koopman operator (Modal structures)
        S_bar (numpy.ndarray): Low-rank approximation of the Koopman operator (Rayleigh quotient matrix).
        xi (numpy.ndarray): Amlitudes of the singular values of the input matrix.

    References:
        [^1]: Schmid, P. (2022). Dynamic Mode Decomposition and Its Variants. 54(1), pp.225-254. doi:[10.1146/annurev-fluid-030121-015835](https://doi.org/10.1146/annurev-fluid-030121-015835).
    """

    def __init__(self, r: int = 0):
        self._C: np.ndarray | None
        self.Y: np.ndarray
        self.K: np.ndarray
        self.Lambda: np.ndarray
        self.m: int
        self.n: int
        self.Phi: np.ndarray
        self.r: int = r
        self.S_bar: np.ndarray
        self._xi: np.ndarray

    @property
    def C(self) -> np.ndarray:
        if self._C is None:
            self._C = np.vander(self.Lambda, self.n, increasing=True)
        return self._C

    @property
    def xi(self) -> np.ndarray:
        if self._xi is None:
            # self._xi = self.Phi.conj().T @ self.Y @ np.linalg.pinv(self.C)
            import cvxpy as cp

            gamma = 0.5
            xi = cp.Variable(self.m)
            objective = cp.Minimize(
                cp.norm(self.Y - self.Phi @ cp.diag(xi) @ self.C, "fro")
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
            problem = cp.Problem(
                objective,
            )

            # Solve the problem
            problem.solve()
        return self._xi

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        # Perform singular value decomposition on X
        u_, sigma, v = np.linalg.svd(X, full_matrices=False)
        r = self.r if self.r > 0 else len(sigma)
        sigma_inv = np.reciprocal(sigma[: r])
        # Compute the low-rank approximation of Koopman matrix
        self.S_bar = (
            u_[: self.m, : r].conj().T
            @ Y
            @ v[: r, :].conj().T
            @ np.diag(sigma_inv)
        )

        # Perform eigenvalue decomposition on S_bar
        self.Lambda, W = np.linalg.eig(self.S_bar)

        # Compute the coefficient matrix
        # TODO: Find out whether to use X or Y (X usage ~ u @ W obviously)
        # self.Phi = X @ v[: r, :].conj().T @ np.diag(sigma_inv) @ W
        self.Phi = u_[:, : r] @ W
        # self.K = self.Phi @ np.diag(self.Lambda) @ np.linalg.pinv(self.Phi)
        self.K = (
            Y
            @ v[: r, :].conj().T
            @ np.diag(sigma_inv)
            @ u_[:, : r].conj().T
        )

    def fit(self, x: np.ndarray):
        """
        Fit the DMD model to the input x.

        Args:
            x: Input x matrix of shape (m, n), where m is the number of variables and n is the number of time steps.

        """
        # Build x matrices
        X = x[:, :-1]
        if hasattr(self, "m"):
            self.Y = x[: self.m, 1:]
        else:
            self.Y = x[:, 1:]

        self.m, self.n = self.Y.shape

        self._fit(X, self.Y)

    def predict(
        self,
        x: np.ndarray,
        forecast: int = 1,
    ) -> np.ndarray:
        """
        Predict future values using the trained DMD model.

        Args:
        forecast: int
            Number of steps to predict into the future.

        Returns:
            predictions: Predicted data matrix for the specified number of prediction steps.
        """
        if self.K is None or self.m is None:
            raise RuntimeError("Fit the model before making predictions.")

        mat = np.zeros((self.m, forecast + 1))
        mat[:, 0] = x
        for s in range(1, forecast + 1):
            mat[:, s] = (self.K @ mat[:, s - 1]).real
        return mat[:, 1:]


class DMDwC(DMD):
    def __init__(self, r: int):
        super().__init__(r)
        self.B: np.ndarray
        self.l: int

    def fit(self, x: np.ndarray, u: np.ndarray, B: np.ndarray | None = None):
        # Need to copy u because it will be modified
        F = u.copy()

        self.l = F.shape[0]
        self.m, self.n = x.shape
        if x.shape[1] != F.shape[1]:
            raise ValueError(
                "x and u must have the same number of time steps.\n"
                f"x: {x.shape[1]}, u: {F.shape[1]}"
            )
        if B is None:
            x = np.vstack((x, F))

            X = x[:, :-1]
            self.Y = x[: self.m, 1:]
        else:
            X = x[:, :-1]
            self.Y = x[:, 1:] - B * F[:, :-1]
        # self.m, self.n = self.Y.shape

        super()._fit(X, self.Y)
        # split K into state transition matrix and control matrix
        self.B = self.K[:, -self.l :]
        self.K = self.K[:, : -self.l]

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
        if self.K is None or self.m is None:
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
            mat[:, s] = (self.K @ mat[:, s - 1]).real + action
        return mat[:, 1:]
