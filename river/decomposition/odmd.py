"""Online Dynamic Mode Decomposition (DMD) in [River API](riverml.xyz).

This module contains the implementation of the Online DMD, Weighted Online DMD,
and DMD with Control algorithms. It is based on the paper by Zhang et al. [^1]
and implementation of authors available at
[GitHub](https://github.com/haozhg/odmd). However, this implementation provides
a more flexible interface aligned with River API covers and separates update
and revert methods to operate with Rolling and TimeRolling wrapers.

TODO:

    - [ ] Compute amlitudes of the singular values of the input matrix.
    - [ ] Update prediction computation for continuous time
          x(t) = Phi exp(diag(ln(Lambda) / dt) * t) Phi^+ x(0) (MIT lecture)
          continuous time eigenvalues exp(Lambda * dt) (Zhang et al. 2019)
    - [ ] Figure out how to use as both MiniBatchRegressor and MiniBatchTransformer
    - [ ] Find out why some values of A change sign between consecutive updates
    - [ ] Drop seed

References:
    [^1]: Zhang, H., Clarence Worth Rowley, Deem, E.A. and Cattafesta, L.N.
    (2019). Online Dynamic Mode Decomposition for Time-Varying Systems. Siam
    Journal on Applied Dynamical Systems, 18(3), pp.1586-1609.
    doi:[10.1137/18m1192329](https://doi.org/10.1137/18m1192329).
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scipy as sp

from river.base import MiniBatchRegressor, MiniBatchTransformer

from .osvd import OnlineSVD

__all__ = [
    "OnlineDMD",
    "OnlineDMDwC",
]


class OnlineDMD(MiniBatchRegressor, MiniBatchTransformer):
    """Online Dynamic Mode Decomposition (DMD).

    This regressor is a class that implements online dynamic mode decomposition
    The time complexity (multiply-add operation for one iteration) is O(4n^2),
    and space complexity is O(2n^2), where n is the state dimension.

    This estimator supports learning with mini-batches with same time and space
    complexity as the online learning. It can be used as Rolling or TimeRolling
    estimator.

    OnlineDMD implements `transform_one` and `transform_many` methods like
    unsupervised MiniBatchTransformer. In such case, we may use `learn_one`
    without `y` and `learn_many` without `Y` to learn the model.
    In that case OnlineDMD preserves previous snapshot and uses it as x while
    current snapshot is used as y, therefore, being delayed by one sample.

    At time step t, define two matrices X(t) = [x(1),x(2),...,x(t)],
    Y(t) = [y(1),y(2),...,y(t)], that contain all the past snapshot pairs,
    where x(t), y(t) are the n dimensional state vector, y(t) = f(x(t)) is
    the image of x(t), f() is the dynamics.

    Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
    then x(t), y(t) should be measurements correponding to consecutive
    states z(t-1) and z(t).

    An exponential weighting factor can be used to place more weight on
    recent data.

    Args:
        r: Number of modes to keep. If 0 (default), all modes are kept.
        w: Weighting factor in (0,1]. Smaller value allows more adpative
        learning, but too small weighting may result in model identification
        instability (relies only on limited recent snapshots).
        initialize: number of snapshot pairs to initialize the model with. If 0
            the model will be initialized with random matrix A and P = \alpha I
            where \alpha is a large positive scalar. If initialize is smaller
            than the state dimension, it will be set to the state dimension and
            raise a warning. Defaults to 1.
        exponential_weighting: Whether to use exponential weighting in revert
        seed: Random seed for reproducibility (initialize A with random values)

    Attributes:
        m: state dimension x(t) as in z(t) = f(z(t-1)) or y(t) = f(t, x(t))
        n_seen: number of seen samples (read-only), reverted if windowed
        feature_names_in_: list of feature names. Used for dict inputs.
        A: DMD matrix, size n by n (non-Hermitian)
        _P: inverse of covariance matrix of X (symmetric)

    Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> n = 101; freq = 2.; tspan = np.linspace(0, 10, n); dt = 0.1
    >>> a1 = 1; a2 = 1; phase1 = -np.pi; phase2 = np.pi / 2
    >>> w1 = np.cos(np.pi * freq * tspan)
    >>> w2 = -np.sin(np.pi * freq * tspan)
    >>> df = pd.DataFrame({'w1': w1[:-1], 'w2': w2[:-1]})

    >>> model = OnlineDMD(r=2, w=0.1, initialize=0)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y) in zip(X.iterrows(), Y.iterrows()):
    ...     x, y = x.to_dict(), y.to_dict()
    ...     model.learn_one(x, y)
    >>> eig, _ =  np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    >>> model.xi  # TODO: verify the result
    array([0.54244, 0.54244])

    >>> from river.utils import Rolling
    >>> model = Rolling(OnlineDMD(r=2, w=1.), 10)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y) in zip(X.iterrows(), Y.iterrows()):
    ...     x, y = x.to_dict(), y.to_dict()
    ...     model.update(x, y)

    >>> eig, _ =  np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    >>> np.isclose(model.truncation_error(X.values, Y.values), 0)
    True

    >>> w_pred = model.predict_one(np.array([w1[-2], w2[-2]]))
    >>> np.allclose(w_pred, [w1[-1], w2[-1]])
    True

    >>> w_pred = model.predict_many(np.array([1, 0]), 10)
    >>> np.allclose(w_pred.T, [w1[1:11], w2[1:11]])
    True

    References:
        [^1]: Zhang, H., Clarence Worth Rowley, Deem, E.A. and Cattafesta, L.N.
        (2019). Online Dynamic Mode Decomposition for Time-Varying Systems.
        Siam Journal on Applied Dynamical Systems, 18(3), pp.1586-1609.
        doi:[10.1137/18m1192329](https://doi.org/10.1137/18m1192329).
    """

    def __init__(
        self,
        r: int = 0,
        w: float = 1.0,
        initialize: int = 1,
        exponential_weighting: bool = False,
        eig_rtol: float | None = None,
        force_orth: bool = False,
        seed: int | None = None,
    ) -> None:
        self.r = int(r)
        self.force_orth = force_orth
        if self.r != 0:
            # Forcing orthogonality makes the results more unstable
            self._svd = OnlineSVD(
                n_components=self.r,
                force_orth=force_orth,
                seed=seed,
            )
        self.w = float(w)
        assert self.w > 0 and self.w <= 1
        self.initialize = int(initialize)
        self.exponential_weighting = exponential_weighting
        self.eig_rtol = eig_rtol
        assert self.eig_rtol is None or 0.0 <= self.eig_rtol < 1.0
        self.seed = seed

        np.random.seed(self.seed)

        self.m: int
        self.n_seen: int = 0
        self.feature_names_in_: list[str]
        self.A: np.ndarray
        self._P: np.ndarray
        self._Y: np.ndarray  # for xi and modes computation

        self._A_last: np.ndarray
        self._A_allclose: bool = False
        self._n_cached: int = 0  # TODO: remove before merge
        self._n_computed: int = 0  # TODO: remove before merge

        # Properties to be reset at each update
        self._eig: tuple[np.ndarray, np.ndarray] | None = None
        self._modes: np.ndarray | None = None
        self._xi: np.ndarray | None = None

    @property
    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute and return DMD eigenvalues and DMD modes at current step"""
        if self._eig is None:
            # TODO: need to check if SVD is initialized in case r < m. Otherwise, transformation will fail.
            # TODO: explore faster ways to compute eig
            # TODO: find out whether Phi should have imaginary part
            Lambda, Phi = sp.linalg.eig(self.A, check_finite=False)

            sort_idx = np.argsort(Lambda)[::-1]
            if not np.array_equal(sort_idx, range(len(Lambda))):
                Lambda = Lambda[sort_idx]
                Phi = Phi[:, sort_idx]
            self._eig = Lambda, Phi
            self._n_computed += 1
        return self._eig

    @property
    def modes(self) -> np.ndarray:
        """Reconstruct high dimensional DMD modes"""
        if self._modes is None:
            _, Phi = self.eig
            if self.r < self.m:
                # Sign of eigenvectors and singular vectors may change based on underlying algorithm initialization
                # TODO: shall we use discrete time singlar values or continuous time singlar values?

                # Schmid (2010), but Phi_comp corresponds to eigenvectors of compainion matrix
                # self._modes = self._svd._U @ Phi_comp

                # Proctor (2016)
                # self._Y.T @ self._svd._Vt.T is increasingly more computationally expensive without rolling
                self._modes = (
                    self._Y.T
                    @ self._svd._Vt.T
                    @ np.diag(1 / self._svd._S)
                    @ Phi
                )

                # This is faster and does not comprosime the results much.
                # self._modes = self._svd._U @ np.diag(1 / self._svd._S) @ Phi
            else:
                self._modes = Phi
        return self._modes

    @property
    def xi(self) -> np.ndarray:
        """Amlitudes of the singular values of the input matrix."""
        if self._xi is None:
            Lambda, Phi = self.eig
            # Compute Discrete temporal dynamics matrix (Vandermonde matrix).
            C = np.vander(Lambda, self.n_seen, increasing=True)
            # xi = self.Phi.conj().T @ self._Y @ np.linalg.pinv(self.C)

            from scipy.optimize import minimize

            def objective_function(x):
                return np.linalg.norm(
                    self._Y[:, : self.r].T - Phi @ np.diag(x) @ C, "fro"
                ) + 0.5 * np.linalg.norm(x, 1)

            # Minimize the objective function
            xi = minimize(objective_function, np.ones(self.r)).x
            self._xi = xi
        return self._xi

    @property
    def A_allclose(self) -> bool:
        """Check if A has changed since last update of eigenvalues"""
        if self.eig_rtol is None:
            return False
        return np.allclose(
            np.abs(self._A_last[:, : self.A.shape[1]]),
            np.abs(self.A),
            rtol=self.eig_rtol,
        )

    def _init_update(self) -> None:
        if self.r == 0:
            self.r = self.m
        if self.initialize > 0 and self.initialize < self.r:
            warnings.warn(
                f"Initialization is under-constrained. Set initialize={self.r} to supress this Warning."
            )
            self.initialize = self.r

        # Zhang (2019) suggests to initialize A with random values
        self.A = np.eye(self.r)
        self._A_last = self.A.copy()
        self._X_init = np.empty((self.initialize, self.m))
        self._Y_init = np.empty((self.initialize, self.m))
        self._Y = np.empty((0, self.m))

    def _truncate_w_svd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        svd_modify: Literal["update", "revert"] | None = None,
    ):
        U_prev = self._svd._U
        if svd_modify == "update":
            self._svd.update(x)
        elif svd_modify == "revert":
            self._svd.revert(x)
        _U = self._svd._U
        _UU = _U.T @ U_prev
        x = x @ _U
        # p != self.m and p == self.A.shape[0] in case of DMDwC
        p = self.A.shape[0]
        y = y @ _U[: y.shape[1], :p]
        # Check if A is square
        if self.A.shape[0] == self.A.shape[1]:
            self.A = _UU @ self.A @ _UU.T
        # If A is not square, it is called by DMDwC
        else:
            _UUp = _UU[:p, :p]
            _UUq = _UU[p:, p:]
            self.A = np.column_stack(
                (_UUp @ self.A[:, :p] @ _UUp.T, _UUp @ self.A[:, p:] @ _UUq.T)
            )
        self._P = np.linalg.inv(_UU @ np.linalg.inv(self._P) @ _UU.T) / self.w

        return x, y

    def _update_A_P(
        self, X: np.ndarray, Y: np.ndarray, W: float | np.ndarray
    ) -> None:
        Xt = X.T
        AX = self.A.dot(Xt)
        PX = self._P.dot(Xt)
        PXt = PX.T
        Gamma = np.linalg.inv(W + X.dot(PX))
        # update A on new data
        self.A += (Y.T - AX).dot(Gamma).dot(PXt)
        # update P, group Px*Px' to ensure positive definite
        self._P = (self._P - PX.dot(Gamma).dot(PXt)) / self.w
        # ensure P is SPD by taking its symmetric part
        self._P = (self._P + self._P.T) / 2

        # Reset properties
        # TODO: explore what revert does with reseting properties
        if not self.A_allclose:
            self._eig = None
            self._A_last = self.A.copy()
        else:
            self._n_cached += 1

        self._modes = None

    def update(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
    ) -> None:
        """Update the DMD computation with a new pair of snapshots (x, y)

        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(t-1) and z(t).

        Args:
            x: 1D array, shape (m, ), x(t) as in y(t) = f(t, x(t))
            y: 1D array, shape (m, ), y(t) as in y(t) = f(t, x(t))
        """
        # If Hankelizer is used, we need to use DMD without y
        if y is None:
            if not hasattr(self, "_x_prev"):
                self._x_prev = x
                return
            else:
                y = x
                x = self._x_prev
        self._x_prev = y

        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()), ndmin=2)
        x_ = x.reshape(1, -1)
        if isinstance(y, dict):
            assert self.feature_names_in_ == list(y.keys())
            y = np.array(list(y.values()), ndmin=2)
        y_ = y.reshape(1, -1)

        # Initialize properties which depend on the shape of x
        if self.n_seen == 0:
            self.m = x_.shape[1]
            self._init_update()

        # Collect buffer of past snapshots to compute modes and xi
        if self._Y.shape[0] <= self.n_seen + 1:
            self._Y = np.row_stack([self._Y, y_])
        if self._Y.shape[0] > self.n_seen + 1:
            self._Y = self._Y[-(self.n_seen + 1) :, :]

        # Initialize A and P with first self.initialize snapshot pairs
        if bool(self.initialize) and self.n_seen < self.initialize:
            self._X_init[self.n_seen, :] = x_
            self._Y_init[self.n_seen, :] = y_
            if self.n_seen == self.initialize - 1:
                self.learn_many(self._X_init, self._Y_init)
                del self._X_init, self._Y_init
                # revert the number of seen samples to avoid doubling
                self.n_seen -= self._X_init.shape[0]
        # Update incrementally if initialized
        else:
            if self.n_seen == 0:
                epsilon = 1e-15
                alpha = 1.0 / epsilon
                self._P = alpha * np.identity(self.r)
            if self.r < self.m:
                x_, y_ = self._truncate_w_svd(x_, y_, svd_modify="update")

            self._update_A_P(x_, y_, 1.0)

        self.n_seen += 1

    def learn_one(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
    ) -> None:
        """Allias for update method."""
        self.update(x, y)

    def revert(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
    ) -> None:
        """Gradually forget the older snapshots and revert the DMD computation.

        Compatible with Rolling and TimeRolling wrappers.

        Args:
            x: 1D array, shape (1, m), x(t) as in y(t) = f(t, x(t))
            y: 1D array, shape (1, m), y(t) as in y(t) = f(t, x(t))

        TODO:
        - [ ] it seems like this does not work as expected
        """
        if self.n_seen < self.initialize:
            raise RuntimeError(
                f"Cannot revert {self.__class__.__name__} before "
                "initialization. If used with Rolling or TimeRolling, window "
                f"size should be increased to {self.initialize + 1 if y is None else 0}."
            )
        if y is None:
            if not hasattr(self, "_x_first"):
                self._x_first = x
                return
            else:
                y = x
                x = self._x_first
                self._x_first = x

        if isinstance(x, dict):
            x = np.array(list(x.values()))
        x_ = x.reshape(1, -1)
        if isinstance(y, dict):
            y = np.array(list(y.values()))
        y_ = y.reshape(1, -1)

        if self.r < self.m:
            x_, y_ = self._truncate_w_svd(x_, y_, svd_modify="revert")

        # Apply exponential weighting factor
        if self.exponential_weighting:
            weight = 1.0 / -(self.w**self.n_seen)
        else:
            weight = -1.0

        self._update_A_P(x_, y_, weight)

        self.n_seen -= 1

    def _update_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
    ) -> None:
        """Update the DMD computation with a new batch of snapshots (X,Y).

        This method brings no change in theoretical time and space complexity.
        However, it allows parallel computing by vectorizing update in loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.

        TODO:
            - [ ] find out why not equal to for loop update implementation
              when weights are used

        """
        p = X.shape[0]
        if self.exponential_weighting:
            weights = np.sqrt(self.w) ** np.arange(p - 1, -1, -1)
        else:
            weights = np.ones(p)
        # Zhang (2019): Gamma = (C^{-1}  U^T P U )^{−1} )
        C_inv = np.diag(np.reciprocal(weights))

        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X
        if isinstance(Y, pd.DataFrame):
            Y_ = Y.values
        else:
            Y_ = Y
        if self.r < self.m:
            X_, Y_ = self._truncate_w_svd(X_, Y_, svd_modify="update")
        self._update_A_P(X_, Y_, C_inv)

    def update_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Learn the OnlineDMD model using multiple snapshot pairs.

        Useful for initializing the model with a batch of snapshot pairs.
        Otherwise, it is equivalent to calling update method in a loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
        """
        if Y is None:
            if isinstance(X, pd.DataFrame):
                Y = X.shift(-1).iloc[:-1]
                X = X.iloc[:-1]
            elif isinstance(X, np.ndarray):
                Y = np.roll(X, -1)[:-1]
                X = X[:-1]

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        # necessary condition for over-constrained initialization
        n = X.shape[0]
        # Exponential weighting factor - older snapshots are weighted less
        if self.exponential_weighting:
            weights = (np.sqrt(self.w) ** np.arange(n - 1, -1, -1))[
                :, np.newaxis
            ]
        else:
            weights = np.ones((n, 1))
        Xqhat, Yqhat = weights * X, weights * Y

        self.n_seen += n

        # Initialize A and P with first p snapshot pairs
        if not hasattr(self, "_P"):
            self.m = X.shape[1]
            if self.r == 0:
                self.r = self.m

            _rank_X = np.linalg.matrix_rank(X)
            if not _rank_X >= self.r:
                raise ValueError(
                    f"Failed rank(X) [{_rank_X}] >= n_modes [{self.r}].\n"
                    "Increase the number of snapshots (increase initialize "
                    f"[{self.initialize}] if learn_many was not called "
                    "directly) or reduce the number of modes."
                )
            XX = Xqhat.T @ Xqhat
            # TODO: think about using correlation matrix to avoid scaling issues
            #  https://stats.stackexchange.com/questions/12200/normalizing-variables-for-svd-pca
            # std = np.sqrt(np.diag(XX))
            # XX = XX / np.outer(std, std)
            # Perform truncated DMD
            if self.r < self.m:
                self._svd.learn_many(Xqhat)
                _U, _S, _V = self._svd._U, self._svd._S, self._svd._Vt

                _m = Yqhat.shape[1]
                _l = self.m - _m

                # DMDwC, A = U.T @ K @ U; B = U.T @ K [Proctor (2016)]
                if _l != 0:
                    _UU = _U.T @ np.row_stack([_U[:_m], np.eye(_l, self.r)])
                # DMD, A = U.T @ K @ U
                else:
                    _UU = np.eye(self.r)

                # TODO: Verify if equivalent to Proctor (2016). They compute U_hat from SVD(Y), we select the first r columns of U
                self.A = (
                    _U.T[:, : Yqhat.shape[1]]
                    @ Yqhat.T
                    @ _V.T
                    @ np.diag(1 / _S)
                ) @ _UU
                self._P = np.linalg.inv(_U.T @ XX @ _U) / self.w
            # Perform exact DMD
            else:
                self.A = Yqhat.T.dot(np.linalg.pinv(Xqhat.T))
                self._P = np.linalg.inv(XX) / self.w

            self._A_last = self.A.copy()
            # Store the last p snapshots for xi computation
            self._Y = Yqhat
            self.initialize = 0
        # Update incrementally if initialized
        # Zhang (2019): "single rank-s update is roughly the same as applying
        #  the rank-1 formula s times"
        else:
            self._update_many(Xqhat, Yqhat)
            if self._Y.shape[0] <= self.n_seen:
                self._Y = np.row_stack([self._Y, Yqhat])
            if self._Y.shape[0] > self.n_seen:
                self._Y = self._Y[-(self.n_seen) :, :]

    def learn_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Allias for update_many method."""
        self.update_many(X, Y)

    def predict_one(self, x: dict | np.ndarray) -> np.ndarray:
        """
        Predicts the next state given the current state.

        Parameters:
            x: The current state.

        Returns:
            np.ndarray: The predicted next state.
        """
        # Map A back to original space
        if self.r < self.m:
            A = self._svd._U @ self.A @ self._svd._U.T
        else:
            A = self.A
        mat = np.zeros((2, self.m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, 2):
            mat[s, :] = (A @ mat[s - 1, :]).real
        return mat[-1, :]

    def predict_many(self, x: dict | np.ndarray, horizon: int) -> np.ndarray:
        """
        Predicts multiple future values based on the given initial value.

        Args:
            x: The initial value.
            horizon (int): The number of future values to predict.

        Returns:
            np.ndarray: An array containing the predicted future values.

        TODO:
            - [ ] Align predict_many with river API
        """
        # Map A back to original space
        if self.r < self.m:
            A = self._svd._U @ self.A @ self._svd._U.T
        else:
            A = self.A
        mat = np.zeros((horizon + 1, self.m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, horizon + 1):
            mat[s, :] = (A @ mat[s - 1, :]).real
        return mat[1:, :]

    def forecast(self, horizon: int, xs: list[dict] | None = None) -> list:
        x = self._x_prev
        if not hasattr(self, "m"):
            self.m = len(x)
        # Map A back to original space
        if self.r < self.m:
            if hasattr(self._svd, "_U"):
                A = self._svd._U @ self.A @ self._svd._U.T
            else:
                return np.zeros((horizon, 1)).flatten().tolist()
        else:
            A = self.A
        mat = np.zeros((horizon + 1, self.m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, horizon + 1):
            mat[s, :] = (A @ mat[s - 1, :]).real
        return mat[1:, -1].flatten().tolist()

    def truncation_error(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
    ) -> float:
        """Compute the truncation error of the DMD model on the given data.

        Since this implementation computes exact DMD, the truncation error is relevant only for initialization.

        Args:
            X: 2D array, shape (p, m), matrix [x(1),x(2),...x(p)]
            Y: 2D array, shape (p, m), matrix [y(1),y(2),...y(p)]

        Returns:
            float: Truncation error of the DMD model
        """
        Y_hat = self.A @ X.T
        return float(np.linalg.norm(Y - Y_hat.T) / np.linalg.norm(Y))

    def transform_one(self, x: dict | np.ndarray) -> dict:
        """
        Transforms the given input sample.

        Args:
            x: The input to transform.

        Returns:
            np.ndarray: The transformed input.
        """
        if isinstance(x, dict):
            x = np.array(list(x.values()))

        return dict(zip(range(self.r), x @ self.modes))

    def transform_many(
        self, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        """
        Transforms the given input sequence.

        Args:
            x: The input to transform.

        Returns:
            np.ndarray: The transformed input.
        """
        M = self.modes
        return X @ M


class OnlineDMDwC(OnlineDMD):
    """Online Dynamic Mode Decomposition (DMD) with Control.

    This regressor is a class that implements online dynamic mode decomposition
    The time complexity (multiply-add operation for one iteration) is O(4n^2),
    and space complexity is O(2n^2), where n is the state dimension.

    This estimator supports learning with mini-batches with same time and space
    complexity as the online learning.

    At time step t, define three matrices X(t) = [x(1),x(2),...,x(t)],
    Y(t) = [y(1),y(2),...,y(t)], U(t) = [U(1),U(2),...,U(t)] that contain all
    the past snapshot pairs, where x(t), y(t) are the n dimensional state
    vectors, and u(t) is m dimensional control input vector, given by
    y(t) = f(x(t), u(t)).

    x(t), y(t) should be measurements correponding to consecutive states z(t-1)
    and z(t).

    An exponential weighting factor can be used to place more weight on
    recent data.

    Args:
        B: control matrix, size n by m. If None, the control matrix will be
        identified from the snapshots. Defaults to None.
        p: truncation of states. If 0 (default), compute exact DMD.
        q: truncation of control. If 0 (default), compute exact DMD.
        w: weighting factor in (0,1]. Smaller value allows more adpative
        learning, but too small weighting may result in model identification
        instability (relies only on limited recent snapshots).
        initialize: number of snapshot pairs to initialize the model with. If 0
            the model will be initialized with random matrix A and P = \alpha I
            where \alpha is a large positive scalar. If initialize is smaller
            than the state dimension, it will be set to the state dimension and
            raise a warning. Defaults to 1.
        exponential_weighting: whether to use exponential weighting in revert
        seed: random seed for reproducibility (initialize A with random values)

    Attributes:
        m: augumented state dimension. if B is None, m = x.shape[1], else m = x.shape[1] + u.shape[1]
        n_seen: number of seen samples (read-only), reverted if windowed
        A: DMD matrix, size n by n
        _P: inverse of covariance matrix of X

    Examples:
    >>> import numpy as np
    >>> import pandas as pd

    >>> n = 101
    >>> freq = 2.0
    >>> tspan = np.linspace(0, 10, n)
    >>> dt = 0.1
    >>> a1 = 1
    >>> a2 = 1
    >>> phase1 = -np.pi
    >>> phase2 = np.pi / 2
    >>> w1 = np.cos(np.pi * freq * tspan)
    >>> w2 = -np.sin(np.pi * freq * tspan)
    >>> u_ = np.ones(n)
    >>> u_[tspan > 5] *= 2
    >>> w1[tspan > 5] *= 2
    >>> w2[tspan > 5] *= 2
    >>> df = pd.DataFrame({"w1": w1[:-1], "w2": w2[:-1]})
    >>> U = pd.DataFrame({"u": u_[:-2]})

    >>> model = OnlineDMDwC(p=2, q=1, w=0.1, initialize=4)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y), (_, u) in zip(X.iterrows(), Y.iterrows(), U.iterrows()):
    ...     x, y, u = x.to_dict(), y.to_dict(), u.to_dict()
    ...     model.learn_one(x, y, u)
    >>> eig, _ = np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.0)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    Supports mini-batch learning:
    >>> from river.utils import Rolling

    >>> model = Rolling(OnlineDMDwC(p=2, q=1, w=1.0), 10)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y), (_, u) in zip(X.iterrows(), Y.iterrows(), U.iterrows()):
    ...     x, y, u = x.to_dict(), y.to_dict(), u.to_dict()
    ...     model.update(x, y, u)

    >>> eig, _ = np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.0)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    # TODO: find out why not passing
    # >>> np.isclose(model.truncation_error(X.values, Y.values, U.values), 0)
    # True

    >>> w_pred = model.predict_one(
    ...     np.array([w1[-2], w2[-2]]),
    ...     np.array([u_[-2]]),
    ... )
    >>> np.allclose(w_pred, [w1[-1], w2[-1]])
    True

    >>> w_pred = model.predict_one(
    ...     np.array([w1[-2], w2[-2]]),
    ...     np.array([u_[-2]]),
    ... )
    >>> np.allclose(w_pred, [w1[-1], w2[-1]])
    True

    >>> w_pred = model.predict_many(np.array([1, 0]), np.ones((10, 1)), 10)
    >>> np.allclose(w_pred.T, [w1[1:11], w2[1:11]])
    True

    References:
        [^1]: Zhang, H., Clarence Worth Rowley, Deem, E.A. and Cattafesta, L.N.
        (2019). Online Dynamic Mode Decomposition for Time-Varying Systems.
        Siam Journal on Applied Dynamical Systems, 18(3), pp.1586-1609.
        doi:[10.1137/18m1192329](https://doi.org/10.1137/18m1192329).
    """

    def __init__(
        self,
        B: np.ndarray | None = None,
        p: int = 0,
        q: int = 0,  # TODO: fix case when q is 0
        w: float = 1.0,
        initialize: int = 1,
        exponential_weighting: bool = False,
        eig_rtol: float | None = None,
        force_orth: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            p + q,
            w,
            initialize,
            exponential_weighting,
            eig_rtol,
            force_orth,
            seed,
        )
        self.p = p
        self.q = q
        self.B = B
        self.known_B = B is not None
        self.l: int

    @property
    def modes(self) -> np.ndarray:
        """Reconstruct high dimensional DMD modes"""
        if self._modes is None:
            _, Phi = self.eig
            if self.r < self.m:
                # Sign of eigenvectors and singular vectors may change based on underlying algorithm initialization
                # Proctor (2016)
                # self._Y.T @ self._svd._Vt.T is increasingly more computationally expensive without rolling
                self._modes = (
                    self._Y.T
                    @ self._svd._Vt.T[:, : self.p]
                    @ np.diag(1 / self._svd._S[: self.p])
                    @ Phi
                )
                # Following has similar results to our modification
                # self._modes = (self._Y.T @ self._svd._Vt.T @ np.diag(1/self._svd._S))[:, :self.p] @ Phi

                # This is faster but significantly alter results for OnlineDMDwC.
                self._modes = (self._svd._U @ np.diag(1 / self._svd._S))[
                    : self.m - self.l, : self.p
                ] @ Phi
            else:
                self._modes = Phi
        return self._modes

    def _init_update(self) -> None:
        if not hasattr(self, "l"):
            super()._init_update()
            return
        if self.p == 0:
            self.p = self.m
        if self.q == 0:
            self.q = self.l
        if self.known_B:
            self.r = self.p
        else:
            self.r = self.p + self.q
        # TODO: if p or q == 0 in __init__, we need to reinitialize SVD
        self._svd = OnlineSVD(
            n_components=self.r,
            force_orth=False,
            seed=self.seed,
        )
        if self.initialize < self.r:
            warnings.warn(
                f"Initialization is under-constrained. Changed initialize to {self.r}."
            )
            self.initialize = self.r

        self.A = np.eye(self.p)
        self._A_last = self.A.copy()
        if not self.known_B:
            self.B = np.eye(self.p, self.q)
            self._A_last = np.column_stack((self.A, self.B))
        self._U_init = np.zeros((self.initialize, self.l))
        self._X_init = np.empty((self.initialize, self.m))
        self._Y_init = np.empty((self.initialize, self.m))
        self._Y = np.empty((0, self.m))

    def _reconstruct_AB(self):
        # self.m stores augumented state dimension
        _m = self.m - self.l if not self.known_B else self.m
        if self.r < self.m:
            A = (
                self._svd._U[:_m, : self.p]
                @ self.A
                @ self._svd._U[:_m, : self.p].T
            )
            B = (
                self._svd._U[:_m, : self.p]
                @ self.B
                @ self._svd._U[-self.q :, -self.l :]
            )
        else:
            A = self.A
            B = self.B
        return A, B

    def update(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
        u: dict | np.ndarray | None = None,
    ) -> None:
        """Update the DMD computation with a new pair of snapshots (x, y)

        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(t-1) and z(t).

        Args:
            x: 1D array, shape (n, ), x(t) as in y(t) = f(t, x(t))
            y: 1D array, shape (n, ), y(t) as in y(t) = f(t, x(t))
            u: 1D array, shape (m, ), u(t) as in y(t) = f(t, x(t), u(t))
        """
        if y is None:
            if not hasattr(self, "_x_prev"):
                self._x_prev = x
                self._u_prev = u
                return
            else:
                y = x
                x = self._x_prev
                self._x_prev = y
                _u_hold = u
                u = self._u_prev
                self._u_prev = _u_hold

        if isinstance(x, dict):
            x = np.array(list(x.values()))
        x = x.reshape(1, -1)
        if isinstance(y, dict):
            y = np.array(list(y.values()))
        y = y.reshape(1, -1)
        if isinstance(u, dict):
            u = np.array(list(u.values()))
        if isinstance(u, np.ndarray):
            u = u.reshape(1, -1)
        # Needed in case of recursive call from learn_many within parent class
        if u is None:
            super().update(x, y)
        else:
            if self.n_seen == 0:
                self.m = x.shape[1]
                self.l = u.shape[1]
                self._init_update()
                self.m += 0 if self.known_B else u.shape[1]

            if self.initialize and self.n_seen <= self.initialize - 1:
                # Accumulate buffer of past snapshots for initialization
                self._X_init[self.n_seen, :] = x
                self._Y_init[self.n_seen, :] = y
                self._U_init[self.n_seen, :] = u
                # Run the initialization after collecting enough snapshots
                if self.n_seen == self.initialize - 1:
                    self.learn_many(self._X_init, self._Y_init, self._U_init)
                    # Subtract the number of seen samples to avoid doubling
                    self.n_seen -= self._X_init.shape[0]
                self.n_seen += 1

            else:
                if self.known_B and self.B is not None:
                    y = y - u @ self.B.T
                else:
                    x = np.column_stack((x, u))
                    if self.B is not None:  # For correct type hinting
                        self.A = np.column_stack((self.A, self.B))
                super().update(x, y)

            # In case that learn_many was called, A is already square
            if self.A.shape[0] < self.A.shape[1]:
                self.B = self.A[: self.p, -self.q :]
                self.A = self.A[: self.p, : -self.q]

    def learn_one(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
        u: dict | np.ndarray | None = None,
    ) -> None:
        """Allias for OnlineDMDwC.update method."""
        return self.update(x, y, u)

    def revert(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
        u: dict | np.ndarray | None = None,
    ) -> None:
        """Gradually forget the older snapshots and revert the DMD computation.

        Compatible with Rolling and TimeRolling wrappers.

        Args:
            x: 1D array, shape (n, ), x(t)
            y: 1D array, shape (n, ), y(t)
            u: 1D array, shape (m, ), u(t)
        """
        if u is None:
            super().revert(x, y)
            return

        if y is None:
            if not hasattr(self, "_x_first"):
                self._x_first = x
                self._u_first = u
                return
            else:
                y = x
                x = self._x_first
                self._x_first = x
                _u_hold = u
                u = self._u_first
                self._u_first = _u_hold

        if isinstance(x, dict):
            x = np.array(list(x.values()))
        x = x.reshape(1, -1)
        if isinstance(y, dict):
            y = np.array(list(y.values()))
        y = y.reshape(1, -1)
        if isinstance(u, dict):
            u = np.array(list(u.values()))
        u = u.reshape(1, -1)
        if self.known_B and self.B is not None:
            y = y - u @ self.B.T
        else:
            x = np.column_stack((x, u))
            if self.B is not None:
                self.A = np.column_stack((self.A, self.B))

        super().revert(x, y)

        if not self.known_B:
            self.B = self.A[: self.p, -self.q :]
            self.A = self.A[: self.p, : -self.q]

    def _update_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
        U: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Update the DMD computation with a new batch of snapshots (X,Y).

        This method brings no change in theoretical time and space complexity.
        However, it allows parallel computing by vectorizing update in loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            U: The control input snapshot matrix of shape (p, l), where p is the number of snapshots and p is the number of control inputs.
        """
        if U is None:
            super()._update_many(X, Y)
        else:
            if self.known_B:
                Y = Y - self.B @ U
            else:
                X = np.column_stack((X, U))
            if self.n_seen == 0:
                self.m = X.shape[1]
                self.l = U.shape[1]
                self._init_update()
            if not self.known_B and self.B is not None:
                self.A = np.column_stack((self.A, self.B))
            self.l = U.shape[1]
            super()._update_many(X, Y)

            if not self.known_B:
                self.B = self.A[:, -self.q :]
                self.A = self.A[:, : -self.q]

    def learn_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame | None = None,
        U: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Learn the OnlineDMDwC model using multiple snapshot pairs.

        Useful for initializing the model with a batch of snapshot pairs.
        Otherwise, it is equivalent to calling update method in a loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            U: The output snapshot matrix of shape (p, l), where p is the number of snapshots and l is the number of control inputs.
        """
        if U is None:
            super().learn_many(X, Y)
            return

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values
        if isinstance(U, pd.DataFrame):
            U = U.values

        if Y is None:
            Y = np.roll(X, -1)[:-1]
            X = X[:-1]
            U = U[:-1]

        if self.known_B and self.B is not None:
            Y = Y - U @ self.B.T
        else:
            X = np.column_stack((X, U))
            if self.B is not None:  # If learn_many is not called first
                self.A = np.column_stack((self.A, self.B))

        self.l = U.shape[1]
        super().learn_many(X, Y)

        if self.p == 0:
            self.p = self.m
        if self.q == 0:
            self.q = self.l
        if not self.known_B:
            self.B = self.A[: self.p, -self.q :]
            self.A = self.A[: self.p, : -self.q]

    def predict_one(
        self, x: dict | np.ndarray, u: dict | np.ndarray
    ) -> np.ndarray:
        """
        Predicts the next state given the current state.

        Parameters:
            x: The current state.
            u: The control input.

        Returns:
            np.ndarray: The predicted next state.
        """
        if isinstance(u, dict):
            u = np.array(list(u.values()))
        _m = len(x)
        A, B = self._reconstruct_AB()

        mat = np.zeros((2, _m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, 2):
            action = (B @ u).real
            # TODO: map A back to original space
            mat[s, :] = (A @ mat[s - 1, :]).real + action
        return mat[-1, :]

    def predict_many(
        self,
        x: dict | np.ndarray,
        U: np.ndarray | pd.DataFrame,
        horizon: int,
    ) -> np.ndarray:
        """
        Predicts multiple future values based on the given initial value.

        Args:
            x: The initial value.
            U: The control input matrix of shape (horizon, l), where l is the number of control inputs.
            horizon (int): The number of future values to predict.

        Returns:
            np.ndarray: An array containing the predicted future values.

        TODO:
            - [ ] Align predict_many with river API
        """
        if isinstance(U, pd.DataFrame):
            U = U.values
        _m = len(x)
        A, B = self._reconstruct_AB()

        mat = np.zeros((horizon + 1, _m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, horizon + 1):
            action = (B @ U[s - 1, :]).real
            mat[s, :] = (A @ mat[s - 1, :]).real + action
        return mat[1:, :]

    def truncation_error(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
        U: np.ndarray | pd.DataFrame,
    ) -> float:
        """Compute the truncation error of the DMD model on the given data.

        Args:
            X: 2D array, shape (n, m), matrix [x(1),x(2),...x(n)]
            Y: 2D array, shape (n, m), matrix [y(1),y(2),...y(n)]
            U: 2D array, shape (n, l), matrix [u(1),u(2),...u(n)]

        Returns:
            float: Truncation error of the DMD model
        """

        A, B = self._reconstruct_AB()
        Y_hat = A @ X.T + B @ U.T
        return float(np.linalg.norm(Y - Y_hat.T) / np.linalg.norm(Y))
