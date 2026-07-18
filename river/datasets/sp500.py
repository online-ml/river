from __future__ import annotations

from river import stream

from . import base

_TICKERS = ["AAPL", "AMZN", "IBM", "INTC", "JNJ", "JPM", "KO", "MSFT", "WMT", "XOM"]


class SP500Stocks(base.FileDataset):
    """Daily returns of ten S&P 500 stocks.

    Daily simple returns of ten large-capitalisation S&P 500 stocks, spanning a diverse set of
    sectors (technology, retail, finance, energy, healthcare, consumer goods): Apple (`AAPL`),
    Amazon (`AMZN`), IBM (`IBM`), Intel (`INTC`), Johnson & Johnson (`JNJ`), JPMorgan (`JPM`),
    Coca-Cola (`KO`), Microsoft (`MSFT`), Walmart (`WMT`) and ExxonMobil (`XOM`). The returns are
    computed from split-adjusted closing prices over February 2013 to February 2018 (1,257 trading
    days), a subset of the popular "S&P 500" stock dataset, and are expressed as percentages
    (a return of 1.0 means +1%).

    Each observation is one trading day: the features are that day's returns for the ten stocks,
    and the target is the *next* trading day's return of an equal-weighted portfolio of the ten
    (a simple market-direction forecasting target). The features on their own are a natural fit
    for the online covariance estimators in `river.covariance`.

    References
    ----------
    [^1]: [S&P 500 stock data (Kaggle)](https://www.kaggle.com/datasets/camnugent/sandp500)

    """

    def __init__(self):
        super().__init__(
            filename="sp500.csv.gz",
            task=base.REG,
            n_features=10,
            n_samples=1_257,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="next_day_return",
            drop=["date"],
            converters={ticker: float for ticker in _TICKERS} | {"next_day_return": float},
        )
