from __future__ import annotations

import collections
import itertools
import math

from river import base, linear_model, preprocessing, time_series

__all__ = ["SNARIMAX"]


class Differencer:
    """A time series differencer.

    References
    ----------
    [^1]: [Stationarity and differencing](https://otexts.com/fpp2/stationarity.html)
    [^2]: [Backshift notation](https://otexts.com/fpp2/backshift.html)

    """

    def __init__(self, d, m=1):
        self.d = d
        self.m = m

        def n_choose_k(n, k):
            f = math.factorial
            return f(n) // f(k) // f(n - k)

        self.coeffs = {0: 1}
        for k in range(1, d + 1):
            t = k * m
            coeff = (-1 if k % 2 else 1) * n_choose_k(n=d, k=k)
            self.coeffs[t] = coeff

    @property
    def n_required_past_values(self):
        return max(self.coeffs)

    @classmethod
    def from_coeffs(cls, coeffs):
        obj = cls(0, 0)
        obj.coeffs = coeffs
        return obj

    def __mul__(self, other):
        """Compose two differencers together."""
        coeffs = collections.defaultdict(int)

        for (t1, c1), (t2, c2) in itertools.product(self.coeffs.items(), other.coeffs.items()):
            coeffs[t1 + t2] += c1 * c2

        # Remove 0 coefficients
        for t, c in list(coeffs.items()):
            if c == 0:
                del coeffs[t]

        return Differencer.from_coeffs(dict(coeffs))

    def diff(self, p, Y: list):
        """Differentiate by applying each coefficient c at each index t.

        Parameters
        ----------
        Y
            The window of previous values. The first element is assumed to be the most recent
            value.

        """
        total = 0
        for t, c in self.coeffs.items():
            try:
                total += (c * Y[t - 1]) if t else p
            except IndexError:
                break
        return total

    def undiff(self, p, Y: list):
        """Differentiate by applying each coefficient c at each index t.

        Parameters
        ----------
        Y
            The window of previous values. The first element is assumed to be the most recent
            value.

        """
        total = p
        for t, c in self.coeffs.items():
            try:
                if t:
                    total -= c * Y[t - 1]
            except IndexError:
                break
        return total


class SNARIMAX(time_series.base.Forecaster):
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

    This model is tailored for time series that are homoskedastic. In other words, it might not
    work well if the variance of the time series varies widely along time.

    Parameters
    ----------
    p
        Order of the autoregressive part. This is the number of past target values that will be
        included as features.
    d
        Differencing order.
    q
        Order of the moving average part. This is the number of past error terms that will be
        included as features.
    m
        Season length used for extracting seasonal features. If you believe your data has a
        seasonal pattern, then set this accordingly. For instance, if the data seems to exhibit
        a yearly seasonality, and that your data is spaced by month, then you should set this
        to 12. Note that for this parameter to have any impact you should also set at least one
        of the `p`, `d`, and `q` parameters.
    sp
        Seasonal order of the autoregressive part. This is the number of past target values
        that will be included as features.
    sd
        Seasonal differencing order.
    sq
        Seasonal order of the moving average part. This is the number of past error terms that
        will be included as features.
    regressor
        The online regression model to use. By default, a `preprocessing.StandardScaler`
        piped with a `linear_model.LinearRegression` will be used.

    Attributes
    ----------
    differencer : Differencer
    y_trues : collections.deque
        The `p` past target values.
    errors : collections.deque
        The `q` past error values.

    Examples
    --------

    >>> import datetime as dt
    >>> from river import datasets
    >>> from river import time_series
    >>> from river import utils

    >>> period = 12
    >>> model = time_series.SNARIMAX(
    ...     p=period,
    ...     d=1,
    ...     q=period,
    ...     m=period,
    ...     sd=1
    ... )

    >>> for t, (x, y) in enumerate(datasets.AirlinePassengers()):
    ...     model = model.learn_one(y)

    >>> horizon = 12
    >>> future = [
    ...     {'month': dt.date(year=1961, month=m, day=1)}
    ...     for m in range(1, horizon + 1)
    ... ]
    >>> forecast = model.forecast(horizon=horizon)
    >>> for x, y_pred in zip(future, forecast):
    ...     print(x['month'], f'{y_pred:.3f}')
    1961-01-01 494.542
    1961-02-01 450.825
    1961-03-01 484.972
    1961-04-01 576.401
    1961-05-01 559.489
    1961-06-01 612.251
    1961-07-01 722.410
    1961-08-01 674.604
    1961-09-01 575.716
    1961-10-01 562.808
    1961-11-01 477.049
    1961-12-01 515.191

    Classic ARIMA models learn solely on the time series values. You can also include features
    built at each step.

    >>> import calendar
    >>> import math
    >>> from river import compose
    >>> from river import linear_model
    >>> from river import optim
    >>> from river import preprocessing

    >>> def get_month_distances(x):
    ...     return {
    ...         calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
    ...         for month in range(1, 13)
    ...     }

    >>> def get_ordinal_date(x):
    ...     return {'ordinal_date': x['month'].toordinal()}

    >>> extract_features = compose.TransformerUnion(
    ...     get_ordinal_date,
    ...     get_month_distances
    ... )

    >>> model = (
    ...     extract_features |
    ...     time_series.SNARIMAX(
    ...         p=1,
    ...         d=0,
    ...         q=0,
    ...         m=12,
    ...         sp=3,
    ...         sq=6,
    ...         regressor=(
    ...             preprocessing.StandardScaler() |
    ...             linear_model.LinearRegression(
    ...                 intercept_init=110,
    ...                 optimizer=optim.SGD(0.01),
    ...                 intercept_lr=0.3
    ...             )
    ...         )
    ...     )
    ... )

    >>> for x, y in datasets.AirlinePassengers():
    ...     model = model.learn_one(x, y)

    >>> forecast = model.forecast(horizon=horizon)
    >>> for x, y_pred in zip(future, forecast):
    ...     print(x['month'], f'{y_pred:.3f}')
    1961-01-01 444.821
    1961-02-01 432.612
    1961-03-01 457.739
    1961-04-01 465.544
    1961-05-01 476.575
    1961-06-01 516.255
    1961-07-01 565.405
    1961-08-01 572.470
    1961-09-01 512.645
    1961-10-01 475.919
    1961-11-01 438.033
    1961-12-01 456.892

    References
    ----------
    [^1]: [ARMA - Wikipedia](https://www.wikiwand.com/en/Autoregressive%E2%80%93moving-average_model)
    [^2]: [NARX - Wikipedia](https://www.wikiwand.com/en/Nonlinear_autoregressive_exogenous_model)
    [^3]: [ARIMA - Forecasting: Principles and Practice](https://otexts.com/fpp2/arima.html)
    [^4]: [Anava, O., Hazan, E., Mannor, S. and Shamir, O., 2013, June. Online learning for time series prediction. In Conference on learning theory (pp. 172-184)](https://arxiv.org/pdf/1302.6927.pdf)

    """

    def __init__(
        self,
        p: int,
        d: int,
        q: int,
        m: int = 1,
        sp: int = 0,
        sd: int = 0,
        sq: int = 0,
        regressor: base.Regressor | None = None,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.m = m
        self.sp = sp
        self.sd = sd
        self.sq = sq
        self.regressor = (
            regressor
            if regressor is not None
            else preprocessing.StandardScaler() | linear_model.LinearRegression()
        )
        self.differencer = Differencer(d=d, m=1) * Differencer(d=sd, m=m)
        self.y_hist: collections.deque[float] = collections.deque(maxlen=d + m * sd)
        self.y_diff: collections.deque[float] = collections.deque(maxlen=max(p, m * sp))
        self.errors: collections.deque[float] = collections.deque(maxlen=max(q, m * sq))

    def _add_lag_features(self, x, Y, errors):
        if x is None:
            x = {}

        # AR
        for t in range(self.p):
            try:
                x[f"y-{t+1}"] = Y[t]
            except IndexError:
                break

        # Seasonal AR
        for t in range(self.m - 1, self.m * self.sp, self.m):
            try:
                x[f"sy-{t+1}"] = Y[t]
            except IndexError:
                break

        # MA
        for t in range(self.q):
            try:
                x[f"e-{t+1}"] = errors[t]
            except IndexError:
                break

        # Seasonal MA
        for t in range(self.m - 1, self.m * self.sq, self.m):
            try:
                x[f"se-{t+1}"] = errors[t]
            except IndexError:
                break

        return x

    def learn_one(self, y, x=None):
        # It isn't possible to difference the time series if enough values have not been seen
        if len(self.y_hist) >= self.differencer.n_required_past_values:
            x = self._add_lag_features(x=x, Y=self.y_diff, errors=self.errors)

            # The regressor learns on the differenced version of the time series
            # If d=0 and/or sd=0, then self.differencer.diff is a no-op
            y_diff = self.differencer.diff(y, self.y_hist)
            self.y_diff.appendleft(y_diff)

            y_pred = self.regressor.predict_one(x)
            self.errors.appendleft(y_diff - y_pred)
            self.regressor.learn_one(x, y_diff)

        self.y_hist.appendleft(y)

        return self

    def forecast(self, horizon, xs=None):
        if xs is None:
            xs = [{}] * horizon

        if len(xs) != horizon:
            raise ValueError("the length of xs should be equal to the specified horizon")

        y_hist = collections.deque(self.y_hist)
        y_diff = collections.deque(self.y_diff)
        errors = collections.deque(self.errors)
        forecasts = [None] * horizon

        for t, x in enumerate(xs):
            x = self._add_lag_features(x=x, Y=y_diff, errors=errors)

            y_pred = self.regressor.predict_one(x)
            y_diff.appendleft(y_pred)

            forecasts[t] = self.differencer.undiff(y_pred, y_hist)
            y_hist.appendleft(forecasts[t])

            errors.appendleft(0)

        return forecasts
