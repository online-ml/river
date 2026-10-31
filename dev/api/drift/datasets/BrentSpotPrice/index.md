# BrentSpotPrice

Brent Spot Price

This is the USD price for Brent Crude oil, measured daily. We include the time series from 2000 onwards. The data is sampled at every 10 original observations to reduce the length of the series.

The data is obtained from the U.S. Energy Information Administration. Since the data is in the public domain, we distribute it as part of this repository.

Since the original data has observations only on trading days, there are arguably gaps in this time series (on non-trading days). However we consider these to be consecutive, and thus also consider the sampled time series to have consecutive observations.

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

- U.S. Energy Information Administration (Sep. 2019)
- https://www.eia.gov/opendata/v1/qb.php?sdid=PET.RBRTE.D
