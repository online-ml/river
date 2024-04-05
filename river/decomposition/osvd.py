"""Online Singular Value Decomposition (SVD) in [River API](riverml.xyz).

This module contains the implementation of the Online SVD algorithm.
It is based on the paper by Brand et al. [^1]

TODO:
    - [ ] Implement update methods based on [2] to save time on reorthogonalization.
    - [ ] Figure out revert method based on [2]

References:
    [^1]: Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415(1), pp.20-30. doi:[10.1016/j.laa.2005.07.021](https://doi.org/10.1016/j.laa.2005.07.021).
    [^2]: Zhang, Y. (2022). An answer to an open question in the incremental SVD. doi:[10.48550/arXiv.2204.05398](https://doi.org/10.48550/arXiv.2204.05398)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy as sp

from river.base import MiniBatchTransformer

__all__ = [
    "OnlineSVD",
    "OnlineSVDZhang",
]


def test_orthonormality(vectors, tol=1e-10):  # pragma: no cover
    """
    Test orthonormality of a set of vectors.

    Parameters:
    vectors : numpy.ndarray
        Matrix where each column represents a vector
    tol : float, optional
        Tolerance for checking orthogonality and unit length

    Returns:
    is_orthonormal : bool
        True if vectors are orthonormal, False otherwise
    """
    # Check unit length
    norms = np.linalg.norm(vectors, axis=0)
    is_unit_length = np.allclose(norms, 1, atol=tol)

    # Check orthogonality
    inner_products = np.dot(vectors.T, vectors)
    off_diagonal = inner_products - np.diag(np.diag(inner_products))
    is_orthogonal = np.allclose(off_diagonal, 0, atol=tol)

    # Check if both conditions are satisfied
    is_orthonormal = is_unit_length and is_orthogonal

    return is_orthonormal


class OnlineSVD(MiniBatchTransformer):
    """Online Singular Value Decomposition (SVD).

    Args:
        n_components: Desired dimensionality of output data. The default value is useful for visualisation.
        initialize: Number of initial samples to use for the initialization of the algorithm. The value must be greater than `n_components`.
        force_orth: If True, the algorithm will force the singular vectors to be orthogonal. *Note*: Significantly increases the computational cost.
        seed: Random seed.

    Attributes:
        n_components: Desired dimensionality of output data.
        initialize: Number of initial samples to use for the initialization of the algorithm. The value must be greater than `n_components`.
        feature_names_in_: List of input features.
        _U: Left singular vectors.
        _S: Singular values.
        _V: Right singular vectors.

    Examples:
    >>> np.random.seed(0)
    >>> r = 3
    >>> m = 4
    >>> n = 80
    >>> X = pd.DataFrame(np.linalg.qr(np.random.rand(n, m))[0])
    >>> svd = OnlineSVD(n_components=r, force_orth=False)
    >>> svd.learn_many(X.iloc[: r * 2])
    >>> svd._U.shape == (m, r), svd._V.shape == (r, r * 2)
    (True, True)

    >>> svd.transform_one(X.iloc[10].to_dict())
    {0: ...0.0494..., 1: ...0.0030..., 2: ...0.0111...}

    >>> for _, x in X.iloc[10:-1].iterrows():
    ...     svd.learn_one(x.values.reshape(1, -1))
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: ...0.0488..., 1: ...0.0613..., 2: ...0.1150...}

    >>> svd.update(X.iloc[-1].to_dict())
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: ...0.0409..., 1: ...0.0336..., 2: ...0.1287...}

    For higher dimensional data and forced orthogonality, revert may not return us to the original state.
    >>> svd.revert(X.iloc[-1].to_dict(), idx=-1)
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: ...0.0488..., 1: ...0.0613..., 2: ...0.1150...}

    >>> svd = OnlineSVD(n_components=0, initialize=3, force_orth=True)
    >>> svd.learn_many(X.iloc[:30])

    >>> svd.learn_many(X.iloc[30:60])
    >>> svd.transform_many(X.iloc[60:62])
               0         1         2         3
    60 ...0.103403  ...0.134656 ...0.108399 ...0.125872
    61 ...0.063485  ...0.023943 ...0.120235 ...0.088502

    References:
    [^1]: Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415(1), pp.20-30. doi:[10.1016/j.laa.2005.07.021](https://doi.org/10.1016/j.laa.2005.07.021).
    """

    def __init__(
        self,
        n_components: int = 2,
        initialize: int = 0,
        force_orth: bool = True,
        seed: int | None = None,
    ):
        self.n_components = n_components
        self.initialize = initialize
        self.force_orth = force_orth
        self.seed = seed

        np.random.seed(self.seed)

        self.n_features_in_: int
        self.feature_names_in_: list
        self._U: np.ndarray
        self._S: np.ndarray
        self._V: np.ndarray

        self.n_seen: int = 0

    @classmethod
    def _from_state(
        cls,
        U: np.ndarray,
        S: np.ndarray,
        V: np.ndarray,
        force_orth: bool = True,
        seed: int | None = None,
    ):
        new = cls(
            n_components=S.shape[0],
            initialize=0,
            force_orth=force_orth,
            seed=seed,
        )
        new._U = U
        new._S = S
        new._V = V
        new.n_seen = V.shape[1]
        return new

    def _orthogonalize(self, U_, Sigma_, V_):
        UQ, UR = np.linalg.qr(U_, mode="complete")
        VQ, VR = np.linalg.qr(V_, mode="complete")
        A = UR @ np.diag(Sigma_) @ VR
        if 0 < self.n_components and self.n_components < min(A.shape):
            tU_, tSigma_, tV_ = sp.sparse.linalg.svds(A, k=self.n_components)
            tU_, tSigma_, tV_ = self._sort_svd(tU_, tSigma_, tV_)
        else:
            tU_, tSigma_, tV_ = np.linalg.svd(A, full_matrices=False)
        return UQ @ tU_, tSigma_, VQ @ tV_

    def _sort_svd(self, U, S, V):
        """Sort the singular value decomposition in descending order.

        As sparse SVD does not guarantee the order of the singular values, we
        need to sort the singular value decomposition in descending order.
        """
        sort_idx = np.argsort(S)[::-1]
        if not np.array_equal(sort_idx, range(len(S))):
            S = S[sort_idx]
            U = U[:, sort_idx]
            V = V[sort_idx, :]
        return U, S, V

    def _truncate_svd(self):
        """Truncate the singular value decomposition to the n components.

        Full SVD returns the full matrices U, S, and V in correct order. If the
        result acqisition is faster than sparse SVD, we combine the results of
        full SVD with truncation.
        """
        self._U = self._U[:, : self.n_components]
        self._S = self._S[: self.n_components]
        self._V = self._V[: self.n_components, :]

    def update(self, x: dict | np.ndarray):
        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()))
        x = x.reshape(1, -1)

        if self.n_seen == 0:
            self.n_features_in_ = x.shape[1]
            if self.n_components == 0:
                self.n_components = self.n_features_in_
            # Make initialize feasible if not set and learn_one is called first
            if not self.initialize:
                self.initialize = self.n_components
            self._X_init = np.empty((self.initialize, self.n_features_in_))
            # Initialize _U with random orthonormal matrix for transform_one
            r_mat = np.random.randn(self.n_features_in_, self.n_components)
            self._U, _ = np.linalg.qr(r_mat)

        # Initialize if called without learn_many
        if bool(self.initialize) and self.n_seen <= self.initialize - 1:
            self._X_init[self.n_seen, :] = x
            if self.n_seen == self.initialize - 1:
                self.learn_many(self._X_init)
                # revert I seen which learn_many accounted for
                self.n_seen -= 1
        else:
            m = (x @ self._U).T
            p = x.T - self._U @ m
            P, _ = np.linalg.qr(p)
            Ra = P.T @ p
            # pad V with zeros to create place for new singular vector
            # TODO: in long term, we may wish to warn about increasing size of V
            _V = np.pad(self._V, ((0, 0), (0, 1)))
            b = np.concatenate([np.zeros(_V.shape[1] - 1), [1]]).reshape(-1, 1)
            n = _V @ b
            q = b - _V.T @ n
            Q, _ = np.linalg.qr(q)

            z = np.zeros_like(m.T)
            K = np.block([[np.diag(self._S), m], [z, Ra]])

            if 0 < self.n_components and self.n_components < min(K.shape):
                U_, Sigma_, V_ = sp.sparse.linalg.svds(K, k=self.n_components)
                U_, Sigma_, V_ = self._sort_svd(U_, Sigma_, V_)
            else:
                U_, Sigma_, V_ = np.linalg.svd(K, full_matrices=False)

            U_ = np.column_stack((self._U, P)) @ U_
            V_ = V_ @ np.row_stack((_V, Q.T))
            # V_ = V_[:, : self.n_components] @ self._V
            if self.force_orth:
                U_, Sigma_, V_ = self._orthogonalize(U_, Sigma_, V_)
            self._U, self._S, self._V = U_, Sigma_, V_

        self.n_seen += 1

    def revert(self, x: dict | np.ndarray, idx: int = 0):

        b = np.zeros(self._V.shape[1])
        b[-1] = 1.0
        b = b.reshape(-1, 1)

        n = self._V[:, idx].reshape(-1, 1)

        q = b - self._V.T @ n
        Q, _ = np.linalg.qr(q)  # Orthonormal basis of column space of q
        # Rb = Q.T @ q
        S_ = np.pad(np.diag(self._S), ((0, 1), (0, 1)))
        # For full-rank SVD, this results in nn == 1.
        nn = n.T @ n
        norm_n = np.sqrt(1.0 - nn) if nn < 1 else 0.0
        K = S_ @ (
            np.identity(S_.shape[0])
            - np.row_stack((n, 0.0)) @ np.row_stack((n, norm_n)).T
        )
        if 0 < self.n_components and self.n_components < min(K.shape):
            U_, Sigma_, V_ = sp.sparse.linalg.svds(K, k=self.n_components)
            U_, Sigma_, V_ = self._sort_svd(U_, Sigma_, V_)
        else:
            U_, Sigma_, V_ = np.linalg.svd(K, full_matrices=False)

        # Since the update is not rank-increasing, we can skip computation of P
        #  otherwise we do U_ = np.column_stack((self._U, P)) @ U_
        U_ = self._U @ U_[: self.n_components, :]

        V_ = V_ @ np.row_stack((self._V, Q.T))[:, :-1]
        # V_ = V_[:, : self.n_components] @ self._V[:, :-1]

        if self.force_orth:  # and not test_orthonormality(U_):
            U_, Sigma_, V_ = self._orthogonalize(U_, Sigma_, V_)
        self._U, self._S, self._V = U_, Sigma_, V_

    def learn_one(self, x: dict | np.ndarray):
        """Allias for update method."""
        self.update(x)

    def learn_many(self, X: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_in_ = [str(i) for i in range(X.shape[0])]

        if self.n_seen == 0:
            self.n_features_in_ = X.shape[1]
            if self.n_components == 0:
                self.n_components = self.n_features_in_

        if hasattr(self, "_U") and hasattr(self, "_S") and hasattr(self, "_V"):
            for x in X:
                self.learn_one(x.reshape(1, -1))
        else:
            assert np.linalg.matrix_rank(X.T) >= self.n_components
            if 0 < self.n_components and self.n_components < min(X.shape):
                self._U, self._S, self._V = sp.sparse.linalg.svds(
                    X.T, k=self.n_components
                )
                self._U, self._S, self._V = self._sort_svd(
                    self._U, self._S, self._V
                )

            else:
                self._U, self._S, self._V = np.linalg.svd(
                    X.T, full_matrices=False
                )
                assert self._S.shape[0] == self.n_components

            self.n_seen = X.shape[0]

    def transform_one(self, x: dict | np.ndarray) -> dict:
        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()))
        else:
            self.feature_names_in_ = [str(i) for i in range(x.shape[0])]

        # If transform one is called before any learning has been done
        # TODO: consider raising an runtime error
        if not hasattr(self, "_U"):
            return dict(
                zip(
                    range(self.n_components),
                    np.zeros(self.n_components),
                )
            )

        return dict(zip(range(self.n_components), x @ self._U))

    def transform_many(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        # If transform one is called before any learning has been done
        # TODO: consider raising an runtime error
        if not hasattr(self, "_U"):
            return pd.DataFrame(
                np.zeros((X.shape[0], self.n_components)),
                index=range(self.n_components),
            )
        assert X.shape[1] == self.n_features_in_

        X_ = X @ self._U
        return pd.DataFrame(X_)


class OnlineSVDZhang(MiniBatchTransformer):
    """Online Singular Value Decomposition (SVD).

    Args:
        n_components: Desired dimensionality of output data. The default value is useful for visualisation.
        initialize: Number of initial samples to use for the initialization of the algorithm. The value must be greater than `n_components`.
        rank_updates: If True, the algorithm will allow rank-increasing updates. *Note*: Significantly increases the computational cost.
        seed: Random seed.

    Attributes:
        n_components: Desired dimensionality of output data.
        initialize: Number of initial samples to use for the initialization of the algorithm. The value must be greater than `n_components`.
        feature_names_in_: List of input features.
        _U: Left singular vectors.
        _S: Singular values.
        _V: Right singular vectors.

    Examples:
    >>> np.random.seed(0)
    >>> r = 3
    >>> m = 4
    >>> n = 80
    >>> X = pd.DataFrame(np.linalg.qr(np.random.rand(n, m))[0])
    >>> svd = OnlineSVDZhang(n_components=r, rank_updates=False)
    >>> svd.learn_many(X.iloc[: r * 2])
    >>> svd._U.shape == (m, r), svd._V.shape == (r, r * 2)
    (True, True)

    >>> svd.transform_one(X.iloc[10].to_dict())
    {0: ...0.0494..., 1: ...0.0030..., 2: ...0.0111...}

    >>> for _, x in X.iloc[10:-1].iterrows():
    ...     svd.learn_one(x.values.reshape(1, -1))
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: ...0.0488..., 1: ...0.0613..., 2: ...0.1150...}

    >>> svd.update(X.iloc[-1].to_dict())
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: ...0.0409..., 1: ...0.0336..., 2: ...0.1287...}

    For higher dimensional data and forced orthogonality, revert may not return us to the original state.
    >>> svd.revert(X.iloc[-1].to_dict(), idx=-1)
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: ...0.0488..., 1: ...0.0613..., 2: ...0.1150...}

    >>> svd = OnlineSVDZhang(n_components=0, initialize=3, rank_updates=False)
    >>> svd.learn_many(X.iloc[:30])

    >>> svd.learn_many(X.iloc[30:60])
    >>> svd.transform_many(X.iloc[60:62])
              0         1         2         3
    60 ...0.103403  ...0.134656 ...0.108399 ...0.125872
    61 ...0.063485  ...0.023943 ...0.120235 ...0.088502

    References:
    [^1]: Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415(1), pp.20-30. doi:[10.1016/j.laa.2005.07.021](https://doi.org/10.1016/j.laa.2005.07.021).
    """

    def __init__(
        self,
        n_components: int = 2,
        initialize: int = 0,
        rank_updates: bool = False,
        seed: int | None = None,
    ):
        self.n_components = n_components
        self.initialize = initialize
        self.rank_updates = rank_updates
        self.seed = seed

        np.random.seed(self.seed)

        self.n_features_in_: int
        self.feature_names_in_: list
        self.n_seen: int = 0
        self._U: np.ndarray
        self._S: np.ndarray
        self._V: np.ndarray

        self.V: np.ndarray
        self.Q0: np.ndarray
        self.q: float = 0.0
        self.W: np.ndarray
        self.tol: float = 1e-15

    @classmethod
    def _from_state(
        cls,
        U: np.ndarray,
        S: np.ndarray,
        V: np.ndarray,
        rank_updates: bool = False,
        seed: int | None = None,
    ):
        new = cls(
            n_components=S.shape[0],
            initialize=0,
            rank_updates=rank_updates,
            seed=seed,
        )
        new.n_features_in_ = U.shape[0]
        new._U = U
        new._S = S
        new._V = V
        new.n_seen = V.shape[1]
        new.V = np.empty((new.n_components, 0))
        new.Q0 = np.identity(new.n_components)
        new.W = np.identity(new.n_features_in_)
        return new

    def _orthogonalize(self, U_, Sigma_, V_):
        UQ, UR = np.linalg.qr(U_, mode="complete")
        VQ, VR = np.linalg.qr(V_, mode="complete")
        A = UR @ np.diag(Sigma_) @ VR
        if 0 < self.n_components and self.n_components < min(A.shape):
            tU_, tSigma_, tV_ = sp.sparse.linalg.svds(A, k=self.n_components)
            tU_, tSigma_, tV_ = self._sort_svd(tU_, tSigma_, tV_)
        else:
            tU_, tSigma_, tV_ = np.linalg.svd(A, full_matrices=False)
        return UQ @ tU_, tSigma_, VQ @ tV_

    def _sort_svd(self, U, S, V):
        """Sort the singular value decomposition in descending order.

        As sparse SVD does not guarantee the order of the singular values, we
        need to sort the singular value decomposition in descending order.
        """
        sort_idx = np.argsort(S)[::-1]
        if not np.array_equal(sort_idx, range(len(S))):
            S = S[sort_idx]
            U = U[:, sort_idx]
            V = V[sort_idx, :]
        return U, S, V

    def _truncate_svd(self):
        """Truncate the singular value decomposition to the n components.

        Full SVD returns the full matrices U, S, and V in correct order. If the
        result acqisition is faster than sparse SVD, we combine the results of
        full SVD with truncation.
        """
        self._U = self._U[:, : self.n_components]
        self._S = self._S[: self.n_components]
        self._V = self._V[: self.n_components, :]

    def update(self, x: dict | np.ndarray):
        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()))
        x = x.reshape(1, -1)

        if self.n_seen == 0:
            self.n_features_in_ = x.shape[1]
            if self.n_components == 0:
                self.n_components = self.n_features_in_
            # Make initialize feasible if not set and learn_one is called first
            if not self.initialize:
                self.initialize = self.n_components
            self._X_init = np.empty((self.initialize, self.n_features_in_))
            # Initialize _U with random orthonormal matrix for transform_one
            r_mat = np.random.randn(self.n_features_in_, self.n_components)
            self._U, _ = np.linalg.qr(r_mat)
            self.V = np.empty((self.n_components, 0))
            self.Q0 = np.identity(self.n_components)
            # TODO: Allow weighting specified by user
            self.W = np.identity(self.n_features_in_)

        # Initialize if called without learn_many
        if bool(self.initialize) and self.n_seen <= self.initialize - 1:
            self._X_init[self.n_seen, :] = x
            if self.n_seen == self.initialize - 1:
                self.learn_many(self._X_init)
                # revert I seen which learn_many accounted for
                self.n_seen -= 1
        else:
            k = self.n_components

            Q, Sigma, R = self._U, self._S, self._V.T  # m x k, k x 1, n x k
            # Step 1: Calculate d, e, p
            d = Q.T @ (self.W @ x.T)  # k x 1
            e = x.T - Q @ d  # m x 1
            p = np.sqrt(e.T @ self.W @ e)
            p[np.isnan(p)] = 1.0  # 1 x 1
            # Step 2: Check tolerance
            if (p < self.tol).all():
                self.q += 1  # 1 x 1
                self.V = np.column_stack((self.V, d))  # k x n_incr
            else:
                if self.q > 0 and self.V.shape[1] > 0:
                    # Step 7: Construct Y
                    Y = np.column_stack(
                        (np.diag(Sigma), self.V)
                    )  # k x k + n_incr
                    # Step 8: Perform SVD on Y
                    QY, SigmaY, RYt = np.linalg.svd(
                        Y, full_matrices=False
                    )  # k x k, k x 1, k x k + n_incr
                    RY = RYt.T  # k + n_incr x k
                    # Step 9: Update Q0, Sigma, R
                    self.Q0 = self.Q0 @ QY  # k x k
                    Sigma = SigmaY  # k x 1
                    _R1 = RY[:k, :-1]  # k x k + n_incr - 1
                    _R2 = RY[k, :-1]  # 1 x k + n_incr - 1
                    R = np.row_stack((R @ _R1, _R2))  # n + 1 x k + n_incr - 1
                    # Step 11: Calculate d
                    d = QY.T @ d  # k x 1
                # Step 13: Normalize e
                e = e @ np.linalg.inv(p)  # m x 1
                # Step 14: Check if |e>W*Q(:, 1)| > tol
                if np.abs(e.T @ (self.W @ Q[:, 0])).any() > self.tol:
                    e = e - Q @ (Q.T @ (self.W @ e))  # m x 1
                    p1 = np.sqrt(e.T @ self.W @ e)  # 1 x 1
                    p1[np.isnan(p1)] = 1.0  # 1 x 1
                    e = e @ np.linalg.inv(p1)  # m x 1
                # Step 17: Construct Y
                Y = np.block(
                    [[np.diag(Sigma), d], [np.zeros_like(d.T), p]]
                )  # k + 1 x k + 1
                QY, SigmaY, RYt = np.linalg.svd(
                    Y
                )  # k + 1 x k + 1, k + 1 x 1, k + 1 x k + 1
                RY = RYt.T  # k + 1 x k + 1
                # Step 20: Update Q0
                Q_0diff = QY.shape[0] - self.Q0.shape[0]
                Q_1diff = QY.shape[1] - self.Q0.shape[1]
                self.Q0 = (
                    np.block(
                        [
                            [self.Q0, np.zeros((self.Q0.shape[0], Q_1diff))],
                            [
                                np.zeros((Q_0diff, self.Q0.shape[1])),
                                np.eye(Q_0diff, Q_1diff),
                            ],
                        ]
                    )
                    @ QY
                )  # k + 1 x k + 1
                Qe = np.column_stack((Q, e))  # m x k + 1
                # TODO: verify implementation of rank increasing updates
                # Step 19: Check if rank increasing
                if SigmaY[k] > self.tol and self.rank_updates:
                    # Step 20 - 21: Update Q, Sigma, R
                    Q = Qe @ self.Q0  # m x k + 1
                    Sigma = SigmaY  # k + 1 x 1
                    _R1 = RY[:k, :]  # k x k + 1
                    _R2 = RY[k, :]  # 1 x k + 1
                    R = np.row_stack((R @ _R1, _R2))  # n + 1 x k + 1
                    self.Q0 = np.eye(k + 1)  # k + 1 x k + 1
                else:
                    # Step 23 - 24: Update Q, Sigma, R
                    Q = Qe @ self.Q0[:, :k]  # m x k
                    Sigma = SigmaY[:k]  # k x 1
                    R_0diff = 1
                    R_1diff = RY.shape[1] - R.shape[1]
                    R = (
                        np.block(
                            [
                                [R, np.zeros((R.shape[0], R_1diff))],
                                [
                                    np.zeros((R_0diff, R.shape[1])),
                                    np.eye(R_0diff, R_1diff),
                                ],
                            ]
                        )
                        @ RY[:, :k]
                    )  # n + 1 x k
                    self.Q0 = np.eye(k)  # k x k

                self.n_components = Sigma.shape[0]
                self.V = np.empty((self.n_components, 0))
                self.q = 0.0

                # Alg. 11
                if self.q > 0:
                    # Step 2: Construct Y
                    Y = np.column_stack((np.diag(Sigma), self.V))
                    # Step 3: Perform SVD on Y
                    QY, SigmaY, RYt = np.linalg.svd(Y, full_matrices=False)
                    RY = RYt.T  # k + 1 x k + 1
                    # Step 4: Update Q, Sigma, R
                    Q = Q @ QY
                    Sigma = SigmaY
                    _R1 = RY[:k, :-1]
                    _R2 = RY[k, :-1]
                    R = np.row_stack((R @ _R1, _R2))
            self._U, self._S, self._V = Q, Sigma, R.T

        self.n_seen += 1

    def revert(self, _: dict | np.ndarray, idx: int = 0):
        # if isinstance(x, dict):
        #     x = np.array(list(x.values()))
        # x = x.reshape(1, -1)

        k = self.n_components
        W = 1.0
        # m = self.n_features_in_

        # n = x.shape[0]
        Q, Sigma, R = self._U, self._S, self._V.T  # m x k, k x 1, n + 1 x k
        # Step 1: Calculate d, e, p
        b = np.zeros(R.shape[0])  # n + 1 x 1
        b[-1] = 1.0
        b = b.reshape(-1, 1)
        d = R.T @ (W * b)  # k x 1
        e = b - R @ d  # n + 1 x 1
        p = np.sqrt(e.T @ (W * e))  # 1 x 1
        p[np.isnan(p)] = 1.0
        if (p < self.tol).all():
            self.q += 1
            self.V = np.column_stack((self.V, d))
        else:
            if self.q > 0 and self.V.shape[1] > 0:
                # Step 7: Construct Y
                Y = np.column_stack((np.diag(Sigma), self.V))
                # Step 8: Perform SVD on Y
                QY, SigmaY, RYt = np.linalg.svd(Y, full_matrices=False)
                RY = RYt.T
                # Step 9: Update Q0, Sigma, R
                self.Q0 = self.Q0 @ QY
                Sigma = SigmaY
                _R1 = RY[:k, :-1]
                _R2 = RY[k, :-1]
                R = np.row_stack((R @ _R1, _R2))
                # Step 11: Calculate d
                d = QY.T @ d
            else:
                self.V = np.column_stack((self.V, d))
            # Step 13: Normalize e
            e = e @ np.linalg.inv(p)
            # Step 14: Check if |e>W*Q(:, 1)| > tol
            if np.abs(e.T @ (W * R[:, 0])).any() > self.tol:
                e = e - R @ (R.T @ (W * e))
                p1 = np.sqrt(e.T @ (W * e))
                p1[np.isnan(p1)] = 1.0
                e = e @ np.linalg.inv(p1)
            # Step 17: Construct Y
            S_ = np.pad(np.diag(Sigma), ((0, 1), (0, 1)))
            # For full-rank SVD, this results in nn == 1.
            nn = d.T @ d
            norm_d = np.sqrt(1.0 - nn) if nn < 1 else 0.0
            Y = S_ @ (
                np.identity(S_.shape[0])
                - np.row_stack((d, 0.0)) @ np.row_stack((d, norm_d)).T
            )
            QY, SigmaY, RYt = np.linalg.svd(Y)
            RY = RYt.T
            # Step 20: Update Q0
            Q_0diff = QY.shape[0] - self.Q0.shape[0]
            Q_1diff = QY.shape[1] - self.Q0.shape[1]
            self.Q0 = (
                np.block(
                    [
                        [self.Q0, np.zeros((self.Q0.shape[0], Q_1diff))],
                        [
                            np.zeros((Q_0diff, self.Q0.shape[1])),
                            np.eye(Q_0diff, Q_1diff),
                        ],
                    ]
                )
                @ QY
            )  # k + 1 x k + 1
            # Step 19: Check if rank decreasing
            if SigmaY[k] < self.tol and self.rank_updates:
                Q = Q @ self.Q0[:k, : k - 1]
                Sigma = SigmaY[: k - 1]
                R = (
                    np.block(
                        [
                            [R, np.zeros((R.shape[0], 1))],
                            [np.zeros((1, R.shape[1])), np.eye(1, 1)],
                        ]
                    )
                    @ RY[:, :k]
                )[2:, : k - 1]
                self.Q0 = np.eye(k - 1)
            else:
                # Step 23 - 24: Update Q, Sigma, R
                Q = Q @ self.Q0[:k, :k]
                Sigma = SigmaY[:k]
                R = (
                    np.block(
                        [
                            [R, np.zeros((R.shape[0], 1))],
                            [np.zeros((1, R.shape[1])), np.eye(1, 1)],
                        ]
                    )
                    @ RY[:, :k]
                )[2:]
                self.Q0 = np.eye(k)

            self.n_components = Sigma.shape[0]
            self.V = np.empty((self.n_components, 0))
            q = 0.0

            # Alg. 11
            if q > 0:
                # Step 2: Construct Y
                Y = np.column_stack((np.diag(Sigma), self.V))
                # Step 3: Perform SVD on Y
                QY, SigmaY, RY = np.linalg.svd(Y, full_matrices=False)
                # Step 4: Update Q, Sigma, R
                Q = Q @ QY
                Sigma = SigmaY
                _R1 = RY[:k, :-1]
                _R2 = RY[k, :-1]
                R = np.row_stack((R @ _R1, _R2))
        self._U, self._S, self._V = Q, Sigma, R.T

        self.n_seen += 1

    def learn_one(self, x: dict | np.ndarray):
        """Allias for update method."""
        self.update(x)

    def learn_many(self, X: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_in_ = [str(i) for i in range(X.shape[0])]

        if self.n_seen == 0:
            self.n_features_in_ = X.shape[1]
            if self.n_components == 0:
                self.n_components = self.n_features_in_
            self.V = np.empty((self.n_components, 0))
            self.Q0 = np.identity(self.n_components)
            # TODO: Allow weighting specified by user
            self.W = np.identity(self.n_features_in_)

        if hasattr(self, "_U") and hasattr(self, "_S") and hasattr(self, "_V"):
            for x in X:
                self.learn_one(x.reshape(1, -1))
        else:
            assert np.linalg.matrix_rank(X.T) >= self.n_components
            if 0 < self.n_components and self.n_components < min(X.shape):
                self._U, self._S, self._V = sp.sparse.linalg.svds(
                    X.T, k=self.n_components
                )
                self._U, self._S, self._V = self._sort_svd(
                    self._U, self._S, self._V
                )

            else:
                self._U, self._S, self._V = np.linalg.svd(
                    X.T, full_matrices=False
                )
                assert self._S.shape[0] == self.n_components

            self.n_seen = X.shape[0]

    def transform_one(self, x: dict | np.ndarray) -> dict:
        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()))
        else:
            self.feature_names_in_ = [str(i) for i in range(x.shape[0])]

        # If transform one is called before any learning has been done
        # TODO: consider raising an runtime error
        if not hasattr(self, "_U"):
            return dict(
                zip(
                    range(self.n_components),
                    np.zeros(self.n_components),
                )
            )

        return dict(zip(range(self.n_components), x @ self._U))

    def transform_many(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        # If transform one is called before any learning has been done
        # TODO: consider raising an runtime error
        if not hasattr(self, "_U"):
            return pd.DataFrame(
                np.zeros((X.shape[0], self.n_components)),
                index=range(self.n_components),
            )
        assert X.shape[1] == self.n_features_in_

        X_ = X @ self._U
        return pd.DataFrame(X_)
