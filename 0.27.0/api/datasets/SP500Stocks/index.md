# SP500Stocks

Daily returns of ten S&amp;P 500 stocks.

Daily simple returns of ten large-capitalisation S&amp;P 500 stocks, spanning a diverse set of sectors (technology, retail, finance, energy, healthcare, consumer goods): Apple (`AAPL`), Amazon (`AMZN`), IBM (`IBM`), Intel (`INTC`), Johnson &amp; Johnson (`JNJ`), JPMorgan (`JPM`), Coca-Cola (`KO`), Microsoft (`MSFT`), Walmart (`WMT`) and ExxonMobil (`XOM`). The returns are computed from split-adjusted closing prices over February 2013 to February 2018 (1,257 trading days), a subset of the popular "S&amp;P 500" stock dataset, and are expressed as percentages (a return of 1.0 means +1%).

Each observation is one trading day: the features are that day's returns for the ten stocks, and the target is the *next* trading day's return of an equal-weighted portfolio of the ten (a simple market-direction forecasting target). The features on their own are a natural fit for the online covariance estimators in `river.covariance`.

## Attributes

- **desc**

  Return the description from the docstring.
- **path**

## Methods

> [!NOTE]
>
> **take**
>
> Iterate over the k samples.
>
> **Parameters**
>
> - **k** — `int`

## References

- [S&amp;P 500 stock data (Kaggle)](<https://www.kaggle.com/datasets/camnugent/sandp500>)
