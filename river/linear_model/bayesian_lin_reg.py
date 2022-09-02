import numpy as np

from river import base, proba, utils


class BayesianLinearRegression(base.Regressor):
    """Bayesian linear regression.

    An advantage of Bayesian linear regression over standard linear regression is that features
    do not have to scaled beforehand. Another attractive property is that this flavor of linear
    regression is somewhat insensitive to its hyperparameters. Finally, this model can output
    instead a predictive distribution rather than just a point estimate.

    The downside is that the learning step runs in `O(n^2)` time, whereas the learning step of
    standard linear regression takes `O(n)` time.

    Parameters
    ----------
    alpha
        Prior parameter.
    beta
        Noise parameter.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics

    >>> dataset = datasets.TrumpApproval()
    >>> model = linear_model.BayesianLinearRegression()
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric).get()
    0.5818

    >>> x, _ = next(iter(dataset))
    >>> model.predict_one(x)
    43.61

    >>> model.predict_one(x, as_dist=True)
    𝒩(μ=43.616, σ=1.003)

    References
    ----------
    [^1]: [Pattern Recognition and Machine Learning, page 52 — Christopher M. Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
    [^2]: [Bayesian/Streaming Algorithms — Vincent Warmerdam](https://koaning.io/posts/bayesian-propto-streaming/)
    [^3]: [Bayesian linear regression for practitioners — Max Halford](https://maxhalford.github.io/blog/bayesian-linear-regression/)

    """

    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self._ss = {}
        self._ss_inv = {}
        self._m = {}
        self._n = 1

    def _unit_test_skips(self):
        return {"check_shuffle_features_no_impact", "check_emerging_features"}

    def _get_arrays(self, features, m=True, ss=True, ss_inv=True):
        m_arr = np.array([self._m.get(i, 0.0) for i in features]) if m else None
        ss_arr = (
            np.array([[self._ss.get(min((i, j), (j, i)), 0.0) for j in features] for i in features])
            if ss
            else None
        )
        ss_inv_arr = (
            np.array(
                [
                    [
                        self._ss_inv.get(min((i, j), (j, i)), 1.0 / self.alpha if i == j else 0.0)
                        for j in features
                    ]
                    for i in features
                ],
                order="F",
            )
            if ss_inv
            else None
        )
        return m_arr, ss_arr, ss_inv_arr

    def _set_arrays(self, features, m_arr, ss_arr, ss_inv_arr):
        for i, fi in enumerate(features):
            self._m[fi] = m_arr[i]
            ss_row = ss_arr[i]
            ss_inv_row = ss_inv_arr[i]
            for j, fj in enumerate(features):
                self._ss[min((fi, fj), (fj, fi))] = ss_row[j]
                self._ss_inv[min((fi, fj), (fj, fi))] = ss_inv_row[j]

    def learn_one(self, x, y):
        x_arr = np.array(list(x.values()))
        m_arr, ss_arr, ss_inv_arr = self._get_arrays(x.keys())

        bx = self.beta * x_arr
        utils.math.sherman_morrison(A=ss_inv_arr, u=bx, v=x_arr)
        # Bishop equation 3.50
        m_arr = ss_inv_arr @ (ss_arr @ m_arr + bx * y)
        # Bishop equation 3.51
        ss_arr += np.outer(bx, x_arr)

        self._set_arrays(x.keys(), m_arr, ss_arr, ss_inv_arr)

        return self

    def predict_one(self, x, as_dist=False):
        """Predict the output of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.
        as_dist
            Whether to return a predictive distribution, or instead just the most likely value.

        Returns
        -------
        The prediction.

        """

        # Bishop equation 3.58
        y_pred_mean = utils.math.dot(self._m, x)
        if not as_dist:
            return y_pred_mean

        x_arr = np.array(list(x.values()))
        _, _, ss_inv_arr = self._get_arrays(x.keys(), m=False, ss=False)
        # Bishop equation 3.59
        y_pred_var = 1 / self.beta + x_arr @ ss_inv_arr @ x_arr.T

        return proba.Gaussian._from_state(n=1, m=y_pred_mean, sig=y_pred_var**0.5, ddof=0)
