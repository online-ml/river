# ClassificationReport

A report for monitoring a classifier.

This class maintains a set of metrics and updates each of them every time `update` is called. You can print this class at any time during a model's lifetime to get a tabular visualization of various metrics.

You can wrap a [`metrics.ClassificationReport`](<https://riverml.xyz/dev/api/metrics/ClassificationReport>) with [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>) in order to obtain a classification report over a window of observations. You can also wrap it with [`utils.TimeRolling`](<https://riverml.xyz/dev/api/utils/TimeRolling>) to obtain a report over a period of time.

## Parameters

- **decimals**

  *Default* → `2`

  The number of decimals to display in each cell.
- **cm**

  *Default* → `None`

  This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage and computation time.

## Attributes

- **bigger\_is\_better**

  Indicate if a high value is better than a low one or not.
- **requires\_labels**

  Indicates if labels are required, rather than probabilities.
- **works\_with\_weights**

  Indicate whether the model takes into consideration the effect of sample weights

## Examples

```python
from river import metrics

y_true = ['pear', 'apple', 'banana', 'banana', 'banana']
y_pred = ['apple', 'pear', 'banana', 'banana', 'apple']

report = metrics.ClassificationReport()

for yt, yp in zip(y_true, y_pred):
    report.update(yt, yp)

print(report)
```

```text
               Precision   Recall   F1       Support
<BLANKLINE>
   apple       0.00%    0.00%    0.00%         1
  banana     100.00%   66.67%   80.00%         3
    pear       0.00%    0.00%    0.00%         1
<BLANKLINE>
   Macro      33.33%   22.22%   26.67%
   Micro      40.00%   40.00%   40.00%
Weighted      60.00%   40.00%   48.00%
<BLANKLINE>
                 40.00% accuracy
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the metric.

> [!NOTE]
>
> **is\_better\_than**
>
> Indicate if the current metric is better than another one.
>
> **Parameters**
>
> - **other**

> [!NOTE]
>
> **revert**
>
> Revert the metric.
>
> **Parameters**
>
> - **y\_true**
> - **y\_pred**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **update**
>
> Update the metric.
>
> **Parameters**
>
> - **y\_true**
> - **y\_pred**
> - **w** — defaults to `1.0`

> [!NOTE]
>
> **works\_with**
>
> Indicates whether or not a metric can work with a given model.
>
> **Parameters**
>
> - **model** — `base.Estimator`
