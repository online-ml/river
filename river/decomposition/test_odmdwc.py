from __future__ import annotations

import numpy as np
import pandas as pd

from river.decomposition.odmd import OnlineDMD, OnlineDMDwC
from river.utils import Rolling

T = 10
t_diff = 0.01
samples = int(T / t_diff) - 1
time_space = np.linspace(0, T, num=samples + 1)


def omega(t):
    return 1 + 0.1 * t


def u_t(x):
    return K_prop * x


X = np.zeros((samples + 1, 2))
X[0, :] = np.array([4, 7])

K_prop = -1

B = np.array([1, 0])
U = np.zeros((samples + 1, 1))

i = 1
true_eigs_ = []
for k in np.linspace(t_diff, T, num=samples):
    A_t = np.array([[t_diff, -omega(k)], [omega(k), 0.1 * t_diff]])
    true_eigs_.append(np.imag(np.log(np.linalg.eig(A_t)[0])))

    control_input = np.matmul(B, u_t(X[i - 1]).T) * t_diff
    U[i, :] = control_input
    autonomous_state = np.matmul(X[i - 1, :], A_t) * t_diff + X[i - 1, :]
    X[i, :] = autonomous_state + control_input
    i += 1

true_eigs = np.vstack(true_eigs_)

X = X[:-1, :]
Y = X[1:, :]
U = U[:-1, :]


def test_input_types():
    n_init = round(samples / 2)

    odmd1 = OnlineDMDwC(initialize=n_init)

    for x, y, u in zip(X, Y, U):
        odmd1.learn_one(x, y, u)

    X_, Y_, U_ = pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(U)

    odmd2 = OnlineDMDwC(initialize=n_init)

    for x, y, u in zip(
        X_.to_dict(orient="records"),
        Y_.to_dict(orient="records"),
        U_.to_dict(orient="records"),
    ):
        odmd2.learn_one(x, y, u)

    assert np.allclose(odmd1.A, odmd2.A)


def test_dmdwc_variations():
    odmd = OnlineDMD(initialize=10)
    odmdc_weight = OnlineDMDwC(
        initialize=10, w=0.995, exponential_weighting=True
    )
    odmdc_b = OnlineDMDwC(initialize=10, B=B.reshape(-1, 1))
    odmdc_window = Rolling(OnlineDMDwC(initialize=10), window_size=100)
    odmdc_b_window = Rolling(
        OnlineDMDwC(initialize=10, B=B.reshape(-1, 1)), window_size=100
    )

    for x_, y_, u_ in zip(X, Y, U):
        odmd.learn_one(x_, y_)
        odmdc_weight.learn_one(x_, y_, u_)
        odmdc_b.learn_one(x_, y_, u_)
        odmdc_window.learn_one(x_, y_, u_)
        odmdc_b_window.learn_one(x_, y_, u_)

    atol = np.abs(get_ct_eigs(odmd.A) - true_eigs[-1]) * 1.5
    eig_weight = get_ct_eigs(odmdc_weight.A)
    assert np.allclose(eig_weight, true_eigs[-1], atol=atol)
    eig_b = get_ct_eigs(odmdc_b.A)
    assert np.allclose(eig_b, true_eigs[-1], atol=atol)
    eig_window = get_ct_eigs(odmdc_window.A)
    assert np.allclose(eig_window, true_eigs[-1], atol=atol)
    eig_b_window = get_ct_eigs(odmdc_b_window.A)
    assert np.allclose(eig_b_window, true_eigs[-1], atol=atol)

def get_ct_eigs(A):
    return np.imag(np.log(np.linalg.eigvals(A))) / t_diff


def test_close_learn_one_learn_many():
    pass
