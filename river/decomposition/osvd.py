"""Online Singular Value Decomposition (SVD) in [River API](riverml.xyz).

This module contains the implementation of the Online SVD algorithm.
It is based on the paper by Brand et al. [^1]

References:
    [^1]: Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415(1), pp.20-30. doi:[10.1016/j.laa.2005.07.021](https://doi.org/10.1016/j.laa.2005.07.021).
    [^2]: Zhang, Y. (2022). An answer to an open question in the incremental SVD. doi:[10.48550/arXiv.2204.05398](https://doi.org/10.48550/arXiv.2204.05398).
    [^3]: Zhang, Y. (2022). A note on incremental SVD: reorthogonalization. doi:[10.48550/arXiv.2204.05398](https://doi.org/10.48550/arXiv.2204.05398).
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


def test_orthonormality(vectors, tol=1e-12):  # pragma: no cover
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


def _orthogonalize(U, S, Vt, tol=1e-12, solver="arpack", random_state=None):
    """Orthogonalize the singular value decomposition.

    This function orthogonalizes the singular value decomposition by performing
    a QR decomposition on the left and right singular vectors.

    TODO: verify if this is the correct way to orthogonalize the SVD.
    [^3]: Zhang, Y. (2022). A note on incremental SVD: reorthogonalization. doi:[10.48550/arXiv.2204.05398](https://doi.org/10.48550/arXiv.2204.05398).
    """
    n_components = S.shape[0]
    # In house implementation of full reorthogonalization
    # UQ, UR = np.linalg.qr(U, mode="complete")
    # VQ, VR = np.linalg.qr(Vt, mode="complete")
    # A = UR @ np.diag(S) @ VR
    # tU, tS, tV = _svd(A, 0, None, solver, random_state)
    # return UQ @ tU_, tSigma_, VQ @ tV_

    # Zhang, Y. (2022)
    if (U[:, -1].T @ U[:, 0] > tol).any():
        for i in range(n_components):
            alpha = U[:, i : i + 1]  # m x 1
            for j in range(i - 1):
                beta = U[:, j]  # m x 1
                U[:, i] = U[:, i] - (alpha.T @ beta) * beta
            norm = np.linalg.norm(U[:, i])
            U[:, i] = U[:, i] / norm
    return U, S, Vt


def _sort_svd(U, S, Vt):
    """Sort the singular value decomposition in descending order.

    As sparse SVD does not guarantee the order of the singular values, we
    need to sort the singular value decomposition in descending order.
    """
    sort_idx = np.argsort(S)[::-1]
    if not np.array_equal(sort_idx, range(len(S))):
        S = S[sort_idx]
        U = U[:, sort_idx]
        Vt = Vt[sort_idx, :]
    return U, S, Vt


def _truncate_svd(U, S, Vt, n_components):
    """Truncate the singular value decomposition to the n components.

    Full SVD returns the full matrices U, S, and V in correct order. If the
    result acqisition is faster than sparse SVD, we combine the results of
    full SVD with truncation.
    """
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]
    return U, S, Vt


def _svd(A, n_components, tol=0.0, v0=None, solver=None, random_state=None):
    """Compute the singular value decomposition of a matrix.

    This function computes the singular value decomposition of a matrix A.
    If n_components < min(A.shape), the function uses sparse SVD for speed up.
    """
    # TODO: sparse is slow if not n_components << min(A.shape)
    #  analyze performance benefits for various differencec between
    #  n_components and min(A.shape)
    if 0 < n_components and n_components < min(A.shape):
        U, S, Vt = sp.sparse.linalg.svds(
            A,
            k=n_components,
            tol=tol,
            v0=v0,
            solver=solver,
            random_state=random_state,
        )
        U, S, Vt = _sort_svd(U, S, Vt)
    else:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # # TODO: implement Optimal truncation if n_components is not set
        # #  Gavish, M., & Donoho, D. L. (2014). The optimal hard threshold for singular values is 4/sqrt(3).
        # beta = A.shape[0] / A.shape[1]
        # omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        # n_components = sum(S > omega)
        U, S, Vt = _truncate_svd(U, S, Vt, n_components)
    return U, S, Vt


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
        _U: Left singular vectors (n_features_in_, n_components).
        _S: Singular values (n_components,).
        _Vt: Right singular vectors (transposed) (n_components, n_seen).

    Examples:
    >>> np.random.seed(0)
    >>> r = 3
    >>> m = 4
    >>> n = 80
    >>> X = pd.DataFrame(np.linalg.qr(np.random.rand(n, m))[0])
    >>> svd = OnlineSVD(n_components=r, force_orth=False)
    >>> svd.learn_many(X.iloc[: r * 2])
    >>> svd._U.shape == (m, r), svd._Vt.shape == (r, r * 2)
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
    >>> svd.transform_many(X.iloc[60:62]).abs()
               0         1         2         3
    60  0.103403   0.134656  0.108399  0.125872
    61  0.063485   0.023943  0.120235  0.088502

    References:
    [^1]: Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415(1), pp.20-30. doi:[10.1016/j.laa.2005.07.021](https://doi.org/10.1016/j.laa.2005.07.021).
    """

    def __init__(
        self,
        n_components: int = 2,
        initialize: int = 0,
        force_orth: bool = True,
        solver="arpack",
        seed: int | None = None,
    ):
        self.n_components = n_components
        self.initialize = initialize
        self.force_orth = force_orth
        self.solver = solver
        self.seed = seed

        np.random.seed(self.seed)

        self.n_features_in_: int
        self.feature_names_in_: list
        self.n_seen: int = 0

        self._U: np.ndarray
        self._S: np.ndarray
        self._Vt: np.ndarray

    @classmethod
    def _from_state(
        cls: type[OnlineSVD],
        U: np.ndarray,
        S: np.ndarray,
        Vt: np.ndarray,
        force_orth: bool = True,
        seed: int | None = None,
    ):
        new = cls(
            n_components=S.shape[0],
            initialize=0,
            force_orth=force_orth,
            seed=seed,
        )
        new.n_features_in_ = U.shape[0]
        new.n_seen = Vt.shape[1]

        new._U = U
        new._S = S
        new._Vt = Vt

        return new

    def _init_first_pass(self, x):
        self.n_features_in_ = x.shape[1]
        if self.n_components == 0:
            self.n_components = self.n_features_in_
        self._X_init = np.empty((0, self.n_features_in_))
        if x.shape[0] == 1:
            # Make initialize feasible if not set and learn_one is called first
            if not self.initialize:
                self.initialize = self.n_components
            # Initialize _U with random orthonormal matrix for transform_one
            r_mat = np.random.randn(self.n_features_in_, self.n_components)
            self._U, _ = np.linalg.qr(r_mat)

    def update(self, x: dict | np.ndarray):
        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()), ndmin=2)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)  # 1 x m

        if self.n_seen == 0:
            self._init_first_pass(x)

        # Initialize if called without learn_many
        if bool(self.initialize) and self.n_seen < self.initialize:
            self._X_init = np.row_stack((self._X_init, x))
            if len(self._X_init) == self.initialize:
                self.learn_many(self._X_init)
                # learn many updated seen, we need to revert last sample which
                #  will be accounted for again at the end of update
                self.n_seen -= x.shape[0]
        else:
            A = x.T  # m x c
            c = A.shape[1]

            Ut = self._U.T  # r x m
            M = Ut @ A  # r x c
            P = A - self._U @ M  # m x c
            #  Results seems to be the same for non rank-increasing updates.
            Po = np.linalg.qr(P)[0]
            Pot = Po.T  # c x m or m x m if m < c
            R_A = Pot @ P  # c x c

            # pad V with zeros to create place for new singular vector
            #  (could be omitted to preserve size of V)
            _Vt = np.pad(self._Vt, ((0, 0), (0, c)))  # r x n + c
            nc = _Vt.shape[1]
            B = np.zeros((nc, c))  # n + c x c
            B[-c:, :] = 1.0
            N = _Vt @ B  # r x c
            V = _Vt.T  # n + c x r
            # Might be less numerically stable
            # VVT = V @ _Vt  # n + c x n + c
            # Q = (np.eye(nc) - VVT) @ B  # n + c x c
            Q = B - V @ N  # n + c x c
            Qot = np.linalg.qr(Q)[0].T  # c x n + c
            # R_B = Q.T @ Q  # c x c

            Z = np.zeros((c, self.n_components))  # c x r
            K = np.block([[np.diag(self._S), M], [Z, R_A]])  # r + c x r + c

            U_, S_, Vt_ = _svd(
                K,
                self.n_components,
                # v0=np.column_stack((self._U, Pot.T))[0,:],  # N > M
                v0=np.row_stack((_Vt, Qot))[:, 0],  # N <= M
                solver=self.solver,
                random_state=self.seed,
            )  # r + c x r; ...; r x r + c

            U_ = np.column_stack((self._U, Po)) @ U_  # m x r
            Vt_ = Vt_ @ np.row_stack((_Vt, Qot))  # r x n + c

            if self.force_orth:
                U_, S_, Vt_ = _orthogonalize(U_, S_, Vt_)

            self._U, self._S, self._Vt = U_, S_, Vt_

        self.n_seen += x.shape[0]

    def revert(self, x: dict | np.ndarray, idx: int = 0):
        c = 1 if isinstance(x, dict) else x.shape[0]
        nc = self._Vt.shape[1]
        B = np.zeros((nc, c))  # n + c x c
        B[-c:] = np.identity(c)
        # Schmid takes first c columns of Vt
        # N = _Vt @ B  # r x c
        if idx >= 0:
            N = self._Vt[:, idx : idx + c]  # r x c
        elif idx == -1:
            N = self._Vt[:, -c:]  # r x c
        else:
            N = self._Vt[:, -c + idx + 1 : idx + 1]  # r x c
        V = self._Vt.T  # n + c x r
        Q = B - V @ N  # n + c x c
        Qot = np.linalg.qr(Q)[
            0
        ].T  # c x n + c; Orthonormal basis of column space of q

        S_ = np.pad(np.diag(self._S), ((0, c), (0, c)))  # r + c x r + c
        # For full-rank SVD, this results in nn == 1.
        NtN = N.T @ N  # c x c
        norm_n = np.sqrt(1.0 - NtN)  # c x c
        norm_n[np.isnan(norm_n)] = 0.0
        K = S_ @ (
            np.identity(S_.shape[0])
            - np.row_stack((N, np.zeros((c, c)))) @ np.row_stack((N, norm_n)).T
        )  # r + c x r + c
        U_, S_, Vt_ = _svd(
            K,
            self.n_components,
            # Seems like this converges to different results
            v0=np.row_stack((self._Vt, Qot))[:, 0],
            solver=self.solver,
            random_state=self.seed,
        )  # r + c x r; ...; r x r + c

        # Since the update is not rank-increasing, we can skip computation of P
        #  otherwise we do U_ = np.column_stack((self._U, P)) @ U_
        U_ = self._U @ U_[: self.n_components, :]  # m x r

        Vt_ = Vt_ @ np.row_stack((self._Vt, Qot))[:, :-c]  # r x n
        # Vt_ = Vt_[:, : self.n_components] @ self._Vt[:, :-c]

        if self.force_orth:  # and not test_orthonormality(U_):
            U_, S_, Vt_ = _orthogonalize(U_, S_, Vt_)

        self._U, self._S, self._Vt = U_, S_, Vt_
        self.n_seen -= c

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
            self._init_first_pass(X)

        if (
            hasattr(self, "_U")
            and hasattr(self, "_S")
            and hasattr(self, "_Vt")
        ):
            if X.shape[0] <= self.n_features_in_:
                self.learn_one(X)
            else:
                for X_part in [
                    X[i : i + self.n_features_in_]
                    for i in range(0, X.shape[0], self.n_features_in_)
                ]:
                    self.learn_one(X_part)

        else:
            assert np.linalg.matrix_rank(X.T) >= self.n_components
            self._U, self._S, self._Vt = _svd(
                X.T,
                self.n_components,
                solver=self.solver,
                random_state=self.seed,
            )

            self.n_seen = X.shape[0]

    def transform_one(self, x: dict | np.ndarray) -> dict:
        if isinstance(x, dict):
            x = np.array(list(x.values()))

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


class OnlineSVDZhang(OnlineSVD):
    """Online Singular Value Decomposition (SVD) using Zhang Algorithm.

    This OnlineSVD implementation handles reorthogonalization and rank-increasing updates automatically.

    Args:
        n_components: Desired dimensionality of output data. The default value is useful for visualisation.
        initialize: Number of initial samples to use for the initialization of the algorithm. The value must be greater than `n_components`.
        rank_updates: If True, the algorithm will allow rank-increasing updates. *Note*: Significantly increases the computational cost.
        seed: Random seed.

    Attributes:
        n_components: Desired dimensionality of output data.
        initialize: Number of initial samples to use for the initialization of the algorithm. The value must be greater than `n_components`.
        feature_names_in_: List of input features.
        _U: Left singular vectors (n_features_in_, n_components).
        _S: Singular values (n_components,).
        _Vt: Right singular vectors (transposed) (n_components, n_seen).

    Examples:
    >>> np.random.seed(0)
    >>> r = 3
    >>> m = 4
    >>> n = 80
    >>> X = pd.DataFrame(np.linalg.qr(np.random.rand(n, m))[0])
    >>> svd = OnlineSVDZhang(n_components=r, rank_updates=False)
    >>> svd.learn_many(X.iloc[: r * 2])
    >>> svd._U.shape == (m, r), svd._Vt.shape == (r, r * 2)
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

    # TODO: Fix the problem related to ongoing batch updates
    >>> svd.transform_many(X.iloc[60:62]).abs()
               0         1         2         3
    60  0.103403  0.134656  0.108399  0.125872
    61  0.063485  0.023943  0.120235  0.088502

    References:
    [^2]: Zhang, Y. (2022). An answer to an open question in the incremental SVD. doi:[10.48550/arXiv.2204.05398](https://doi.org/10.48550/arXiv.2204.05398).
    """

    def __init__(
        self,
        n_components: int = 2,
        initialize: int = 0,
        tol: float = 1e-12,
        rank_updates: bool = False,
        seed: int | None = None,
    ):
        super().__init__(
            n_components=n_components,
            initialize=initialize,
            force_orth=False,
            seed=seed,
        )
        self.tol: float = tol
        self.rank_updates = rank_updates

        self._V_buff: np.ndarray
        self._U0: np.ndarray
        self._q_u: int = 0
        self._q_r: int = 0
        self.W: np.ndarray

    @classmethod
    def _from_state(
        cls: type[OnlineSVDZhang],
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
        new.n_seen = V.shape[1]

        new._U = U
        new._S = S
        new._Vt = V

        new._V_buff = np.empty((new.n_components, 0))
        new._U0 = np.identity(new.n_components)
        new.W = np.identity(new.n_features_in_)

        return new

    def _init_first_pass(self, x):
        super()._init_first_pass(x)
        self._V_buff = np.empty((self.n_components, 0))
        self._U0 = np.identity(self.n_components)
        # TODO: Allow weighting specified by user
        self.W = np.identity(self.n_features_in_)

    def update(self, x: dict | np.ndarray):
        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()), ndmin=2)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        if self.n_seen == 0:
            self._init_first_pass(x)

        c = x.shape[0]

        # Initialize if called without learn_many
        if bool(self.initialize) and self.n_seen < self.initialize:
            self._X_init = np.row_stack((self._X_init, x))
            if len(self._X_init) == self.initialize:
                self.learn_many(self._X_init)
                # learn many updated seen, we need to revert last sample which
                #  will be accounted for again at the end of update
                self.n_seen -= c
        else:
            if c > 1:
                from warnings import warn

                warn(
                    "Calling update/learn_many with batches provides different results than incrementing one sample at the time."
                )
            r = self.n_components
            A = x.T  # m x c
            _U, _S, _V = self._U, self._S, self._Vt.T  # m x r, r x 1, n x r
            _Ut = _U.T  # r x m
            # Step 1: Calculate d, e, p
            M = _Ut @ (self.W @ A)  # r x c
            P = A - _U @ M  # m x c
            PtP = P.T @ self.W @ P  # c x c
            PtP_cond = (PtP < 0.0).any()
            if PtP_cond:
                # Approx. 2x slower more stable solution for batched updates
                Po = np.linalg.qr(P)[0]
                Pot = Po.T  # c x m
                Ra = Pot @ P  # c x c
            else:
                Ra = np.sqrt(PtP)  # c x c
            # Step 2: Check tolerance
            if (Ra < self.tol).all():  # n_incr += c
                self._q_u += c  # 1 x 1
                self._V_buff = np.column_stack((self._V_buff, M))  # r x n_incr
            else:
                if self._q_u > 0:
                    # Step 7: Construct Y
                    Y = np.column_stack(
                        (np.diag(_S), self._V_buff)
                    )  # r x r + n_incr
                    # Step 8: Perform SVD on Y
                    UY, SY, VYt = np.linalg.svd(
                        Y, full_matrices=False
                    )  # r x r, r x 1, r x r + n_incr
                    VY = VYt.T  # r + n_incr x r
                    # Step 9: Update U0, _S, _V
                    self._U0 = self._U0 @ UY  # r x r
                    _S = SY  # r x 1
                    _V1 = VY[:r, :-1]  # r x r + n_incr - 1
                    _V2 = VY[r, :-1]  # 1 x r + n_incr - 1
                    _V = np.row_stack(
                        (_V @ _V1, _V2)
                    )  # n + 1 x r + n_incr - 1
                    # Step 11: Calculate d
                    M = UY.T @ M  # r x c
                # Step 13: Normalize e
                if not PtP_cond:
                    Po = P @ np.linalg.inv(Ra)  # m x c
                    Pot = Po.T  # c x m
                # Step 14: Reorthogonalize if |e>W*_U(:, 1)| > tol
                if (np.abs(Pot @ (self.W @ _U[:, 0])) > self.tol).any():
                    Po = Po - _U @ (_Ut @ (self.W @ Po))  # m x c
                    Po = np.linalg.qr(Po)[0]
                # Step 17: Construct Y
                Y = np.block(
                    [
                        [np.diag(_S), M],
                        [np.zeros((c, r)), Ra],
                    ]
                )  # r + c x r + c
                # Not using sp.sparse.linalg.svds for non-rank increasing
                #  updates as it is slower than np.linalg.svd
                UY, SY, VYt = np.linalg.svd(
                    Y
                )  # r + c x r + c, r + c x 1, r + c x r + c
                VY = VYt.T  # r + c x r + c
                # Step 20: Update U0
                self._U0 = (
                    np.block(
                        [
                            [
                                self._U0,
                                np.zeros((self._U0.shape[0], c)),
                            ],
                            [
                                np.zeros((c, self._U0.shape[1])),
                                np.eye(c, c),
                            ],
                        ]
                    )
                    @ UY
                )  # r + c x r + c
                _Ue = np.column_stack((_U, Po))  # m x k + c
                # Step 19: Check if rank increasing
                if self.rank_updates and SY[r] > self.tol:
                    # Step 20 - 21: Update _U, _S, _V
                    _U = _Ue @ self._U0  # m x r + c
                    _S = SY  # r + c x c
                    _V1 = VY[:r, :]  # r x r + c
                    _V2 = VY[r, :]  # 1 x r + c
                    _V = np.row_stack((_V @ _V1, _V2))  # n + 1 x r + 1
                    self._U0 = np.eye(r + 1)  # r + 1 x r + 1
                else:
                    # Step 23 - 24: Update _U, _S, _V
                    _U = _Ue @ self._U0[:, :r]  # m x r
                    _S = SY[:r]  # r x 1
                    V_1pad = VY.shape[1] - _V.shape[1]
                    _V = (
                        np.block(
                            [
                                [_V, np.zeros((_V.shape[0], V_1pad))],
                                [
                                    np.zeros((c, _V.shape[1])),
                                    np.eye(c, V_1pad),
                                ],
                            ]
                        )
                        @ VY[:, :r]
                    )  # n + 1 x r
                    self._U0 = np.eye(r)  # r x r

                # Alg. 11
                #  We note that the output of Algorithm 7 (11), V , may be not empty. This implies that the output of Algorithm 7 (11) is not the SVD of U. Hence we have to update the SVD for the vectors in V
                # This step adds rows to _V to account for the ones buffered in V
                if self._q_u > 0 and self._V_buff.shape[1] > 0:
                    # Step 2: Construct Y
                    Y = np.column_stack(
                        (np.diag(_S), self._V_buff)
                    )  # r x r + v_cols
                    # Step 3: Perform SVD on Y
                    UY, SY, VYt = np.linalg.svd(Y, full_matrices=False)
                    VY = VYt.T  # r + 1 x r + 1
                    # Step 4: Update _U, _S, _V
                    _U = _U @ UY
                    _S = SY
                    _V1 = VY[:r, :]
                    _V2 = VY[r : r + self._q_u + c - 1, :]
                    _V = np.row_stack((_V @ _V1, _V2))

                self.n_components = _S.shape[0]
                self._V_buff = np.empty((self.n_components, 0))
                self._q_u = 0
                self._U, self._S, self._Vt = _U, _S, _V.T

        self.n_seen += c

    def revert(self, x: dict | np.ndarray, idx: int = 0):
        _U, _S, _V = self._U, self._S, self._Vt.T  # m x r, r x 1, n + c x r

        c = 1 if isinstance(x, dict) else x.shape[0]
        nc = self._Vt.shape[1]
        # Step 1: Calculate N, Q, Qot
        B = np.zeros((nc, c))  # n + c x c
        B[-c:] = np.identity(c)

        if idx >= 0:
            N = self._Vt[:, idx : idx + c]  # r x c
        elif idx == -1:
            N = self._Vt[:, -c:]  # r x c
        else:
            N = self._Vt[:, -c + idx + 1 : idx + 1]  # r x c
        Q = B - _V @ N  # n + c x c
        QtQ = Q.T @ Q
        QtQ_cond = (QtQ < 0.0).any()
        if QtQ_cond:
            Ra = np.linalg.qr(Q)[0].T @ Q  # c x c
        else:
            Ra = np.sqrt(Q.T @ Q)  # c x c
        # TODO: not activated at all, check why
        if Ra.size > 0 and (Ra < self.tol).all():
            self._q_r += c
        else:
            if self._q_r > 0:
                c += self._q_r
                B = np.zeros((nc, c))  # n + c x c
                B[-c:] = np.identity(c)
                if idx >= 0:
                    N = self._Vt[:, idx : idx + c]  # r x c
                elif idx == -1:
                    N = self._Vt[:, -c:]  # r x c
                else:
                    N = self._Vt[:, -c + idx + 1 : idx + 1]  # r x c
                Q = B - _V @ N  # n + c x c
            # Step 13: Normalize Q
            Qot = np.linalg.qr(Q)[
                0
            ].T  # c x n + c; Orthonormal basis of column space of q
            # We do not touch original U therefore we leave reorthogonalization to update method :)

            S_ = np.pad(np.diag(_S), ((0, c), (0, c)))  # r + c x r + c
            # For full-rank SVD, this results in nn == 1.
            NtN = N.T @ N  # c x c
            norm_n = np.sqrt(1.0 - NtN)  # c x c
            norm_n[np.isnan(norm_n)] = 0.0
            K = S_ @ (
                np.identity(S_.shape[0])
                - np.row_stack((N, np.zeros((c, c))))
                @ np.row_stack((N, norm_n)).T
            )  # r + c x r + c
            U_, S_, Vt_ = np.linalg.svd(K, full_matrices=False)

            if self.rank_updates and S_[-1] <= self.tol:
                self.n_components -= 1
            U_ = _U @ U_[: self.n_components, : self.n_components]  # m x r
            S_ = S_[: self.n_components]
            Vt_ = (
                Vt_[: self.n_components, :]
                @ np.row_stack((self._Vt, Qot))[:, :-c]
            )  # r x n

            self._q_r = 0
            self._U, self._S, self._Vt = U_, S_, Vt_

        self.n_seen -= c
