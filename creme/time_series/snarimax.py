import collections
import itertools
import math

from .. import linear_model
from .. import preprocessing

from . import base


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
            raise ValueError('m must be greater than or equal to 1')

        self.coeffs = make_coeffs(d=d, m=m)

    def __add__(self, other):
        """Composes two differencers together.

        Example:

            >>> differencer = Differencer(d=3, m=2) + Differencer(d=3, m=1)
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
        return y + sum(
            c * y_previous[t]
            for t, c in self.coeffs.items()
            if t < len(y_previous)
        )

    def undiff(self, y, y_previous):
        """Undifferentiates a value.

            y (float): The value to differentiate.
            y_previous (list of float): The window of previous values. The first element is assumed
                to be the most recent value.

        """
        return y - sum(
            c * y_previous[t]
            for t, c in self.coeffs.items()
            if t < len(y_previous)
        )


class SNARIMAX(base.Forecaster):
    """SNARIMAX model.

    SNARIMAX stands for (S)easonal (N)on-linear (A)uto(R)egressive (I)ntegrated (M)oving-(A)verage
    with e(X)ogenous inputs model.

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
        m (int): Season length used for extracting seasonal features. If you believe your data has
            a seasonal pattern, then set this accordingly. For instance, if the data seems to
            exhibit a yearly seasonality, and that your data is spaced by month, then you should
            set this to 12. Note that for this parameter to have any impact you should also set at
            least one of the P, D, and Q parameters.
        sp (int): Seasonal order of the autoregressive part. This is the number of past target
            values that will be included as features.
        sd (int): Seasonal differencing order.
        sq (int): Seasonal order of the moving average part. This is the number of past error terms
            that will be included as features.
        regressor (base.Regressor): The online regression model to use. By default, a
            ``preprocessing.StandardScaler`` piped with a ``linear_model.LinearRegression`` will
            be used.

    Attributes:
        differencer (Differencer)
        y_trues (collections.deque): The ``p`` past target values.
        errors (collections.deque): The ``q`` past error values.

    Example:

        ::

            >>> import calendar
            >>> import datetime as dt
            >>> from creme import compose
            >>> from creme import datasets
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import time_series

            >>> def get_month_distances(x):
            ...     return {
            ...         calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
            ...         for month in range(1, 13)
            ...     }

            >>> def get_ordinal_date(x):
            ...     return {'ordinal_date': x['month'].toordinal()}

            >>> extract_features = compose.TransformerUnion([
            ...     get_ordinal_date,
            ...     get_month_distances
            ... ])

            >>> model = (
            ...     extract_features |
            ...     time_series.SNARIMAX(
            ...         p=0,
            ...         d=0,
            ...         q=0,
            ...         m=12,
            ...         sp=3,
            ...         sq=6,
            ...         regressor=(
            ...             preprocessing.StandardScaler() |
            ...             linear_model.LinearRegression(
            ...                 intercept=110,
            ...                 optimizer=optim.SGD(0.01),
            ...                 intercept_lr=0.3
            ...             )
            ...         )
            ...     )
            ... )

            >>> metric = metrics.Rolling(metrics.MAE(), 12)

            >>> for x, y in datasets.Airline():
            ...     y_pred = model.fit_predict_one(x=x, y=y)
            ...     metric = metric.update(y, y_pred)

            >>> metric
            Rolling of size 12 MAE: 11.770681

            >>> horizon = 12
            >>> future = [
            ...     {'month': dt.date(year=1961, month=m, day=1)}
            ...     for m in range(1, horizon + 1)
            ... ]
            >>> forecast = model.forecast(horizon=horizon, xs=future)
            >>> for x, y_pred in zip(future, forecast):
            ...     print(x['month'], f'{y_pred:.3f}')
            1961-01-01 441.663
            1961-02-01 426.288
            1961-03-01 470.272
            1961-04-01 481.677
            1961-05-01 487.235
            1961-06-01 538.953
            1961-07-01 625.014
            1961-08-01 622.069
            1961-09-01 521.439
            1961-10-01 449.184
            1961-11-01 398.737
            1961-12-01 431.324

    Note:
        This model is tailored for time series that are homoskedastic. In other words, it might not
        work well if the variance of the time series varies widely along time.

    References:
        1. `Wikipedia page on ARMA <https://www.wikiwand.com/en/Autoregressive%E2%80%93moving-average_model>`_
        2. `Wikipedia page on NARX <https://www.wikiwand.com/en/Nonlinear_autoregressive_exogenous_model>`_
        3. `ARIMA models <https://otexts.com/fpp2/arima.html>`_
        4. `Online Learning for Time Series Prediction <https://arxiv.org/pdf/1302.6927.pdf>`_

    """

    def __init__(self, p, d, q, m=1, sp=0, sd=0, sq=0, regressor=None):
        self.p = p
        self.d = d
        self.q = q
        self.m = m
        self.sp = sp
        self.sd = sd
        self.sq = sq
        self.regressor = (
            regressor if regressor is not None else
            preprocessing.StandardScaler() | linear_model.LinearRegression()
        )
        self.differencer = Differencer(d=d, m=1) + Differencer(d=sd, m=1)
        self.y_trues = collections.deque(maxlen=max(p, m * sp))
        self.errors = collections.deque(maxlen=max(p, m * sq))

    def _add_lag_features(self, x, y_trues, errors):

        if x is None:
            x = {}

        # AR
        for t in range(self.p):
            try:
                x[f'y-{t+1}'] = y_trues[t]
            except IndexError:
                break

        # Seasonal AR
        for t in range(self.m - 1, (self.m - 1) * self.sp, self.m):
            try:
                x[f'sy-{t+1}'] = y_trues[t]
            except IndexError:
                break

        # MA
        for t in range(self.q):
            try:
                x[f'e-{t+1}'] = errors[t]
            except IndexError:
                break

        # Seasonal MA
        for t in range(self.m - 1, (self.m - 1) * self.sq, self.m):
            try:
                x[f'se-{t+1}'] = errors[t]
            except IndexError:
                break

        return x

    def fit_predict_one(self, y, x=None):
        """Updates the model and returns the prediction for the next time step.

        Parameters:
            x (dict): Optional additional features to learn from. In the litterature these are called the
                exogenous variables.
            y (float): In the litterature this is called the endogenous variable.

        """

        # Check there are enough observations so that differencing can happen
        y = self.differencer.diff(y=y, y_previous=self.y_trues)
        x = self._add_lag_features(x=x, y_trues=self.y_trues, errors=self.errors)
        y_pred = self.regressor.predict_one(x)
        self.regressor.fit_one(x, y)

        self.y_trues.appendleft(y)
        self.errors.appendleft(y - y_pred)

        return y_pred

    def fit_one(self, y, x=None):
        """Updates the model.

        Parameters:
            x (dict): Optional additional features to learn from. In the litterature these are called the
                exogenous variables.
            y (float): In the litterature this is called the endogenous variable.

        """

        self.fit_predict_one(y=y, x=x)
        return self

    def forecast(self, horizon, xs=None):

        if xs is None:
            xs = [{}] * horizon

        if len(xs) != horizon:
            raise ValueError('the length of xs should be equal to the specified horizon')

        y_trues = collections.deque(self.y_trues)
        errors = collections.deque(self.errors)
        forecasts = [None] * horizon

        for t, x in enumerate(xs):
            x = self._add_lag_features(x=x, y_trues=y_trues, errors=errors)
            y_pred = self.regressor.predict_one(x)
            forecasts[t] = self.differencer.undiff(y=y_pred, y_previous=y_trues)

            y_trues.appendleft(y_pred)
            errors.appendleft(0)

        return forecasts
