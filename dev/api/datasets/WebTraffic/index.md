# WebTraffic

Web sessions information from an events company based in South Africa.

The goal is to predict the number of web sessions in 4 different regions in South Africa.

The data consists of 15 minute interval traffic values between '2023-06-16 00:00:00' and '2023-09-15 23:45:00' for each region. Two types of sessions are captured `sessionsA` and `sessionsB`. The `isMissing` flag is equal to 1 if any of the servers failed to capture sessions, otherwise if all servers functioned properly this flag is equal to 0.

Things to consider:

- region `R5` captures sessions in backup mode. Strictly speaking, `R5` is not necessary to predict. \* Can `sessionsA` and `sessionsB` events be predicted accurately for each region over the next day (next 96 intervals)? \* What is the best way to deal with the missing values? \* How can model selection be used (a multi-model approach)? \* Can dependence (correlation) between regions be utilised for more accurate predictions? \* Can both `sessionA` and `sessionB` be predicted simultaneously with one model?

This dataset is well suited for time series forecasting models, as well as anomaly detection methods. Ideally, the goal is to build a time series forecasting model that is robust to the anomalous events and generalise well on normal operating conditions.

## Attributes

- **desc**

  Return the description from the docstring.
- **is\_downloaded**

  Indicate whether or the data has been correctly downloaded.
- **path**

## Methods

> [!NOTE]
>
> **download**

> [!NOTE]
>
> **take**
>
> Iterate over the k samples.
>
> **Parameters**
>
> - **k** — `int`
