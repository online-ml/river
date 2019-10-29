import collections
import itertools
import math


__all__ = ['SNARIMAX']


def make_coeffs(d, m):
    """Precomputes the coefficients of the backshift operator.

    Example:

        >>> make_coeffs(1, 1)
        {0: -1}

        >>> make_coeffs(2, 1)
        {0: -2, 1: 1}

        >>> make_coeffs(3, 1)
        {0: -3, 1: 3, 2: -1}

        >>> make_coeffs(2, 7)
        {6: -2, 13: 1}

    """

    def n_choose_k(n, k):
        f = math.factorial
        return f(n) // f(k) // f(n - k)

    return dict(
        (
            k * m - 1,
            int(math.copysign(1, (k + 1) % 2 - 1)) * n_choose_k(n=d, k=k)
        )
        for k in range(1, d + 1)
    )


class Differencer:
    """A time series differencer.

    Example:

        >>> differencer = Differencer(2); differencer.coeffs
        {0: -2, 1: 1}

        >>> differencer.diff(7, [3, 1])
        2

        >>> differencer.undiff(2, [3, 1])
        7

    References:
        1. `Stationarity and differencing <https://otexts.com/fpp2/stationarity.html>`_

    """

    def __init__(self, d, m=1):

        if d < 0:
            raise ValueError('d must be greater than or equal to 0')

        if m < 1:
            raise ValueError('d must be greater than or equal to 1')

        self.coeffs = make_coeffs(d=d, m=m)

    def __mul__(self, other):
        """Composes two differencers together.

        Example:

            >>> differencer = Differencer(d=3, m=2) * Differencer(d=3, m=1)
            >>> for t, c in sorted(differencer.coeffs.items()):
            ...     print(t, c)
            0 -3
            2 8
            3 -6
            4 -6
            5 8
            7 -3
            8 1

        References:
            1. `Backshift notation <https://otexts.com/fpp2/backshift.html>`_

        """
        coeffs = collections.Counter()
        coeffs.update(self.coeffs)
        coeffs.update(other.coeffs)

        for (t1, c1), (t2, c2) in itertools.product(self.coeffs.items(), other.coeffs.items()):
            coeffs[t1 + t2 + 1] += c1 * c2

        # Remove 0 coefficients
        for t in list(coeffs.keys()):
            if coeffs[t] == 0:
                del coeffs[t]

        differencer = Differencer(0, 1)
        differencer.coeffs = dict(coeffs)
        return differencer

    def diff(self, y, y_previous):
        """Differentiates a value.

            y (float): The value to differentiate.
            y_previous (list of float): The window of previous values. The first element is assumed
                to be the most recent value.

        """
        return y + sum(c * y_previous[t] for t, c in self.coeffs.items())

    def undiff(self, y, y_previous):
        """Undifferentiates a value.

            y (float): The value to differentiate.
            y_previous (list of float): The window of previous values. The first element is assumed
                to be the most recent value.

        """
        return y - sum(c * y_previous[t] for t, c in self.coeffs.items())


class SNARIMAX:
    """SNARIMAX model.

    SNARIMAX stands for (S)easonal (N)on-linear (A)uto(R)egressive (I)ntegrated (M)oving-(A)verage
    with e(X)ogenousinputs model.

    This model generalizes many established time series models in a single interface that can be
    trained online. It assumes that the provided training data is ordered in time and is uniformly
    spaced. It is made up of the following components:

    - S (Seasonal)
    - N (Non-linear): Any online regression model can be used, not necessarily a linear regression
        as is done in textbooks.
    - AR (Autoregressive): Lags of the target variable are used as features.
    - I (Integrated): The model can be fitted on a differenced version of a time series. In this
        context, integration is the reverse of differencing.
    - MA (Moving average): Lags of the errors are used as features.
    - X (Exogenous): Users can provide additional features. Care has to be taken to include
        features that will be available both at training and prediction time.

    Each of these components can be switched on and off by specifying the appropriate parameters.
    Classical time series models such as AR, MA, ARMA, and ARIMA can thus be seen as special
    parametrizations of the SNARIMAX model.

    Parameters:
        p (int): Order of the autoregressive part. This is the number of past target values that
            will be included as features.
        d (int): Differencing order.
        q (int): Order of the moving average part. This is the number of past error terms that will
            be included as features.
        s (int):
        sp (int)
        sd (int): Seasonal difference order.
        sq (int)
        regressor (base.Regressor): The online regression model to use. By default, a
            `StandardScaler` piped with a `LinearRegression` will be used.

    Attributes:
        differencer (Differencer)

    Note:
        This model is tailored for time series that are homoskedastic. In other words, it might not work
        very if the variance of the time series varies widely along time.

    References:
        1. https://www.wikiwand.com/en/Autoregressive%E2%80%93moving-average_model
        2. https://www.wikiwand.com/en/Nonlinear_autoregressive_exogenous_model
        3. `ARIMA models <https://otexts.com/fpp2/arima.html>`_

    To do:
        1. Predict ahead
        2. Differencing
        3. Seasonal differencing
        4. Box-Cox transform

    """

    def __init__(self, p, d, q, regressor):

        self.p = p
        self.d = d
        self.q = q
        self.regressor = regressor
        self.differencer = Differencer(d=d)
        self.y_trues = collections.deque(maxlen=max(p, q))
        self.y_preds = collections.deque(maxlen=q)

    def _make_features(self, x, y_trues, y_preds):

        if x is None:
            x = {}

        for t in range(min(self.p, len(self.y_trues))):
            x[f'y-{t+1}'] = self.y_trues[t]

        for t in range(min(self.q, len(self.y_preds))):
            x[f'e-{t+1}'] = self.y_trues[t] - self.y_preds[t]

        return x

    def fit_one(self, y, x=None):
        """

            x (dict): Optional additional features to learn from. In the litterature these are called the
                exogenous variables.
            y (float): In the litterature this is called the endogenous variable.

        """

        x = self._make_features(x=x, y_trues=self.y_trues, y_preds=self.y_preds)
        y_pred = self.regressor.predict_one(x)
        self.regressor.fit_one(x, y)

        self.y_trues.appendleft(y)
        self.y_preds.appendleft(y_pred)

        return self

    def forecast(self, horizon, xs=None):
        """
        Parameters:
            horizon (int): The number of steps ahead to forecast.
            xs (list): The set of . If given, then it's length should be equal to horizon.

        """

        if xs is None:
            xs = [{}] * horizon

        if len(xs) != horizon:
            raise ValueError('the length of xs should be equal to the specified horizon')

        y_trues = collections.deque(self.y_trues)
        y_preds = collections.deque(self.y_preds)
        forecasts = [None] * horizon

        for t, x in enumerate(xs):
            x = self._make_features(x=x, y_trues=self.y_trues, y_preds=self.y_preds)
            y_pred = self.regressor.predict_one(x)
            forecasts[t] = y_pred

            y_trues.appendleft(y_pred)
            y_preds.appendleft(y_pred)

        return forecasts
