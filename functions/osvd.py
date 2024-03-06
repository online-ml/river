# -*- coding: utf-8 -*-
"""Online Singular Value Decomposition (SVD) in [River API](riverml.xyz).

This module contains the implementation of the Online SVD algorithm.
It is based on the paper by Brand et al. [^1]

References:
    [^1]: Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415(1), pp.20-30. doi:[10.1016/j.laa.2005.07.021](https://doi.org/10.1016/j.laa.2005.07.021).
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import scipy as sp
from river.base import MiniBatchTransformer

__all__ = [
    "OnlineSVD",
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
        force_orth: If True, the algorithm will force the singular vectors to be orthogonal. *Note*: Significantly increases the computational cost.

    Attributes:
        n_components_: Desired dimensionality of output data.
        initialize: Number of initial samples to use for the initialization of the algorithm. The value must be greater than `n_components`.
        feature_names_in_: List of input features.
        _U: Left singular vectors.
        _S: Singular values.
        _V: Right singular vectors.

    Examples:
    >>> np.random.seed(0)
    >>> m = 20
    >>> n = 80
    >>> X = pd.DataFrame(np.random.rand(n, m))
    >>> svd = OnlineSVD(n_components=2, force_orth=True)
    >>> svd.learn_many(X.iloc[:10])
    >>> svd._U.shape == (m, 2)
    True
    >>> svd.transform_one(X.iloc[10].to_dict())
    {0: 0.2588, 1: -1.9574}
    >>> for _, x in X.iloc[10:-1].iterrows():
    ...     svd.learn_one(x.values.reshape(1, -1))
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: 2.5420, 1: 0.05388}

    >>> svd.update(X.iloc[-1].values.reshape(1, -1))
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: 2.3492, 1: 0.03840}

    >>> svd.revert(X.iloc[-1].values.reshape(1, -1))

    TODO: fix revert method - following test should pass
    >>> svd.transform_one(X.iloc[0].to_dict())
    {0: 2.3492, 1: 0.03840}

    Works with mini-batches as well
    >>> svd = OnlineSVD(n_components=2, initialize=3, force_orth=True)
    >>> svd.learn_many(X.iloc[:30])
    >>> svd.learn_many(X.iloc[30:60])
    >>> svd.transform_many(X.iloc[60:62])
              0         1
    0  0.103185 -2.409013
    1 -0.066338 -1.896232

    References:
    [^1]: Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415(1), pp.20-30. doi:[10.1016/j.laa.2005.07.021](https://doi.org/10.1016/j.laa.2005.07.021).
    """

    def __init__(
        self,
        n_components: int = 2,
        initialize: int = 0,
        force_orth: bool = False,
    ):
        self.n_components_ = n_components
        if initialize <= n_components:
            self.initialize = n_components + 1
        else:
            self.initialize = initialize
        self.force_orth_ = force_orth
        self.n_features_in_: int
        self.feature_names_in_: list
        self._U: np.ndarray
        self._S: np.ndarray
        self._V: np.ndarray

    def _orthogonalize(self, U_, Sigma_, V_):
        UQ, UR = np.linalg.qr(U_, mode="complete")
        VQ, VR = np.linalg.qr(V_, mode="complete")
        tU_, tSigma_, tV_ = sp.sparse.linalg.svds(
            (UR @ np.diag(Sigma_) @ VR), k=2
        )
        return UQ @ tU_, tSigma_, VQ @ tV_

    def update(self, x: Union[dict, np.ndarray]):
        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()))
        m = (x @ self._U).T
        p = x.T - self._U @ m
        P, _ = np.linalg.qr(p)
        Ra = P.T @ p
        z = np.zeros_like(m.T)
        K = np.block([[np.diag(self._S), m], [z, Ra]])
        U_, Sigma_, V_ = sp.sparse.linalg.svds(K, k=self.n_components_)
        U_ = np.column_stack((self._U, P)) @ U_
        V_ = V_[:, :2] @ self._V
        if self.force_orth_ and not test_orthonormality(V_.T):
            U_, Sigma_, V_ = self._orthogonalize(U_, Sigma_, V_)
        self._U, self._S, self._V = U_, Sigma_, V_

    def revert(self, _: Union[dict, np.ndarray]):
        # TODO: verify proper implementation of revert method
        b = np.concatenate([np.zeros(self._V.shape[1] - 1), [1]]).reshape(
            -1, 1
        )
        n = self._V @ b
        q = b - self._V.T @ n
        Q, _ = np.linalg.qr(q)
        # Rb = Q.T @ q
        S_ = np.pad(np.diag(self._S), ((0, 1), (0, 1)))
        K = S_ @ (
            np.identity(S_.shape[0])
            - np.row_stack((np.diag(self._S) @ n, 0.0))
            @ np.row_stack((n, np.sqrt(1 - n.T @ n))).T
        )
        U_, Sigma_, V_ = sp.sparse.linalg.svds(K, k=2)
        U_ = self._U @ U_[:2, :]
        V_ = V_ @ np.row_stack((self._V, Q.T))

        if self.force_orth_ and not test_orthonormality(U_):
            U_, Sigma_, V_ = self._orthogonalize(U_, Sigma_, V_)
        self._U, self._S, self._V = U_, Sigma_, V_

    def learn_one(self, x: Union[dict, np.ndarray]):
        """Allias for update method."""
        self.update(x)

    def learn_many(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        self.n_features_in_ = X.shape[1]

        if hasattr(self, "_U") and hasattr(self, "_S") and hasattr(self, "_V"):
            for x in X:
                self.learn_one(x.reshape(1, -1))
        else:
            self._U, self._S, self._V = sp.sparse.linalg.svds(
                X.T, k=self.n_components_
            )

    def transform_one(
        self, x: Union[dict, np.ndarray]
    ) -> Union[dict, np.ndarray]:
        is_dict = isinstance(x, dict)
        if is_dict:
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()))

        x_ = self._U.T @ x.T
        return x_ if not is_dict else dict(zip(self.feature_names_in_, x_))

    def transform_many(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            self.feature_names_in_ = list(X.columns)
            X = X.values
        assert X.shape[1] == self.n_features_in_

        X_ = self._U.T @ X.T
        return X_.T if not is_df else pd.DataFrame(X_.T)
