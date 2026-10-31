# MultiLabelConfusionMatrix

Multi-label confusion matrix.

Under the hood, this stores one [`metrics.ConfusionMatrix`](<https://riverml.xyz/dev/api/metrics/ConfusionMatrix>) for each output.

## Examples

```python
from river import metrics

cm = metrics.multioutput.MultiLabelConfusionMatrix()

y_true = [
    {0: False, 1: True, 2: True},
    {0: True, 1: True, 2: False}
]

y_pred = [
    {0: True, 1: True, 2: True},
    {0: True, 1: False, 2: False}
]

for yt, yp in zip(y_true, y_pred):
    cm.update(yt, yp)

cm
```

```text
0
            False   True
    False       0      1
     True       0      1
<BLANKLINE>
1
            False   True
    False       0      0
     True       1      1
<BLANKLINE>
2
            False   True
    False       1      0
     True       0      1
```

## Methods

> [!NOTE]
>
> **revert**

> [!NOTE]
>
> **update**
