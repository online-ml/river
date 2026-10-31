# LeveragingBaggingClassifier

Leveraging Bagging ensemble classifier.

Leveraging Bagging [^/dev/api/ensemble/LeveragingBaggingClassifier/#1] is an improvement over the Oza Bagging algorithm. The bagging performance is leveraged by increasing the re-sampling. It uses a poisson distribution to simulate the re-sampling process. To increase re-sampling it uses a higher `w` value of the Poisson distribution (agerage number of events), 6 by default, increasing the input space diversity, by attributing a different range of weights to the data samples.

To deal with concept drift, Leveraging Bagging uses the ADWIN algorithm to monitor the performance of each member of the enemble If concept drift is detected, the worst member of the ensemble (based on the error estimation by ADWIN) is replaced by a new (empty) classifier.

## Parameters

- **model**

  *Type* → `base.Classifier`

  The classifier to bag.
- **n\_models**

  *Type* → `int`

  *Default* → `10`

  The number of models in the ensemble.
- **w**

  *Type* → `float`

  *Default* → `6`

  Indicates the average number of events. This is the lambda parameter of the Poisson distribution used to compute the re-sampling weight.
- **adwin\_delta**

  *Type* → `float`

  *Default* → `0.002`

  The delta parameter for the ADWIN change detector.
- **bagging\_method**

  *Type* → `str`

  *Default* → `bag`

  The bagging method to use. Can be one of the following:  
  \* 'bag' - Leveraging Bagging using ADWIN.  
  \* 'me' - Assigns \\(weight=1\\) if sample is misclassified, otherwise \\(weight=error/(1-error)\\).  
  \* 'half' - Use resampling without replacement for half of the instances.  
  \* 'wt' - Resample without taking out all instances.  
  \* 'subag' - Resampling without replacement.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **bagging\_methods**

  Valid bagging\_method options.
- **models**

## Examples

```python
from river import datasets
from river import ensemble
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()

model = ensemble.LeveragingBaggingClassifier(
    model=(
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression()
    ),
    n_models=3,
    seed=42
)

metric = metrics.F1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
F1: 88.55%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x**
> - **y**
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `base.typing.ClfTarget | None`: The predicted label.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Averages the predictions of each classifier.
>
> **Parameters**
>
> - **x**
> - **kwargs**

[^/dev/api/ensemble/LeveragingBaggingClassifier/#1]: Bifet, A., Holmes, G., &amp; Pfahringer, B. (2010). Leveraging Bagging for Evolving Data Streams. In Joint European conference on machine learning and knowledge discovery in databases (pp. 135-150). Springer, Berlin, Heidelberg.
