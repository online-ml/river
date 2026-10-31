# WaterFlow

Water flow through a pipeline branch.

The series includes hourly values for about 2 months, March 2022 to May 2022. The values are expressed in liters per second. There are four anomalous segments in the series:

- 3 "low value moments": this is due to water losses or human intervention for maintenance \* A small peak in the water inflow after the first 2 segments: this is due to a pumping operation into the main pipeline, when more water pressure is needed

This dataset is well suited for time series forecasting models, as well as anomaly detection methods. Ideally, the goal is to build a time series forecasting model that is robust to the anomalous segments.

This data has been kindly donated by the Tecnojest s.r.l. company (www.invidea.it) from Italy.

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
