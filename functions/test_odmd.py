# -*- coding: utf-8 -*-
"""Test conversion from river to scikit-learn API and back.

Requires two modifications to river code:
1. change line 49 in river.compat.river_to_sklearn to
`SKLEARN_INPUT_Y_PARAMS = {"multi_output": True, "y_numeric": False}`
2. change line 194 in river.compat.river_to_sklearn to
`y_pred = np.empty(shape=(len(X), X.shape[1]))`
"""

import os
import sys

import numpy as np
from scipy.integrate import odeint

# from river.compat.river_to_sklearn import convert_river_to_sklearn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.dmd import DMD  # noqa: E402
from functions.odmd import OnlineDMD  # noqa: E402

epsilon = 1e-1


def dyn(x, t):
    x1, x2 = x
    dxdt = [(1 + epsilon * t) * x2, -(1 + epsilon * t) * x1]
    return dxdt


# integrate from initial condition [1,0]
samples = 101
tspan = np.linspace(0, 10, samples)
dt = 0.1
x0 = [1, 0]
xsol = odeint(dyn, x0, tspan).T
# extract snapshots
X, Y = xsol[:, :-1], xsol[:, 1:]
t = tspan[1:]
n, m = X.shape
A = np.empty((n, n, m))
eigvals = np.empty((n, m), dtype=complex)
for k in range(m):
    A[:, :, k] = np.array(
        [[0, (1 + epsilon * t[k])], [-(1 + epsilon * t[k]), 0]]
    )
    eigvals[:, k] = np.linalg.eigvals(A[:, :, k])


def test_allclose_online_batch():
    dmd = DMD()
    odmd = OnlineDMD()

    dmd.fit(X, Y)

    odmd = OnlineDMD()
    eigvals_online_ = np.empty((n, m), dtype=complex)
    for i, (x, y) in enumerate(zip(X.T, Y.T)):
        odmd.learn_one(x, y)
        eigvals_online_[:, i] = np.log(np.linalg.eigvals(odmd.A)) / dt

    eigvals_batch = np.log(np.linalg.eigvals(dmd.A)) / dt
    eigvals_online = np.log(np.linalg.eigvals(odmd.A)) / dt

    assert np.allclose(eigvals_batch, eigvals_online)


def test_allclose_weighted_true():
    init_samples = round(samples / 2)
    odmd = OnlineDMD(w=0.9)
    # odmd.learn_many(X[:, :init_samples], Y[:, :init_samples])

    eigvals_online_ = np.empty((n, m), dtype=complex)
    for i, (x, y) in enumerate(zip(X.T, Y.T)):
        odmd.learn_one(x, y)
        eigvals_online_[:, i] = np.log(np.linalg.eigvals(odmd.A)) / dt

    slope_eig_true = np.diff(eigvals)[0, init_samples:].mean()
    slope_eig_online = np.diff(eigvals_online_)[0, init_samples:].mean()
    print(slope_eig_true, slope_eig_online)
    np.allclose(
        slope_eig_true,
        slope_eig_online,
        atol=1e-4,
    )


# def test_conversion():
#     try:
#         dmd = DMD()
#         odmd = OnlineDMD()
#         dmd_sk = convert_river_to_sklearn(odmd)

#         omega = lambda t: 1 + 0.1 * t  # noqa: E731
#         x_0 = np.array([1, 0])
#         X = [x_0]
#         t_diff = 0.1
#         for i in np.linspace(0, 10, num=100):
#             A_t = np.array([[0, omega(i)], [-omega(i), 0]])
#             x_t = np.matmul(X[-1], A_t) * t_diff + X[-1]
#             X.append(x_t)
#         X = np.vstack(X)

#         dmd.fit(X.T[:, :-2])

#         dmd_sk.fit(X.T[:, :-2].T, X.T[:, 1:-1].T)

#         odmd = OnlineDMD()
#         for x, y in zip(X.T[:, :-2].T, X.T[:, 1:-1].T):
#             odmd.learn_one(x, y)

#         y_gt = X.T[:, -1]
#         y_pred_batch = dmd.predict(X.T[:, -2])
#         y_pred_sk = dmd_sk.predict(X.T[:, -2].reshape(1, -1))
#         y_pred_online = odmd.predict_one(X.T[:, -2])

#         assert np.allclose(y_pred_sk, y_pred_online)
#         assert np.allclose(y_pred_sk, y_pred_batch)
#     except AssertionError as e:
#         print("Batch prediction error: ", np.linalg.norm(y_gt - y_pred_batch))
#         print("Online prediction error: ", np.linalg.norm(y_gt - y_pred_online))
#         print("Sklearn prediction error: ", np.linalg.norm(y_gt - y_pred_sk))
#         raise e
