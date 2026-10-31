# ChebyshevOverSampler

Over-sampling for imbalanced regression using Chebyshev's inequality.

Chebyshev's inequality can be used to define the probability of target observations being frequent values (w.r.t. the distribution mean).

Let \\(Y\\) be a random variable with finite expected value \\(\\overline{y}\\) and non-zero variance \\(\\sigma^2\\). For any real number \\(t \> 0\\), the Chebyshev's inequality states that, for a wide class of unimodal probability distributions: \\(Pr(|y-\\overline{y}| \\ge t\\sigma) \\le \\dfrac{1}{t^2}\\).

Taking \\(t=\\dfrac{|y-\\overline{y}|}{\\sigma}\\), and assuming \\(t \> 1\\), the Chebyshev’s inequality for an observation \\(y\\) becomes: \\(P(|y - \\overline{y}|=t) = \\dfrac{\\sigma^2}{|y-\\overline{y}|}\\).

Alternatively, one can use \\(t\\) directly to estimate a frequency weight \\(\\kappa = \\lceil t\\rceil\\) and define an over-sampling strategy for extreme and rare target values[^/dev/api/imblearn/ChebyshevOverSampler/#1]. Each incoming instance is used \\(\\kappa\\) times to update the underlying regressor. Frequent target values contribute only once to the underlying regressor, whereas rares cases are used multiple times for training.

## Parameters

- **regressor**

  *Type* → `base.Regressor`

  The regression model that will receive the biased sample.

## Examples

```python
from river import datasets
from river import evaluate
from river import imblearn
from river import metrics
from river import preprocessing
from river import rules

model = (
    preprocessing.StandardScaler() |
    imblearn.ChebyshevOverSampler(
        regressor=rules.AMRules(
            n_min=50, delta=0.01
        )
    )
)

evaluate.progressive_val_score(
    datasets.TrumpApproval(),
    model,
    metrics.MAE(),
    print_every=500
)
```

```text
[500] MAE: 1.629786
[1,000] MAE: 1.663799
[1,001] MAE: 1.66253
MAE: 1.66253
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Fits to a set of features `x` and a real-valued target `y`.
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
> Predict the output of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The prediction.

[^/dev/api/imblearn/ChebyshevOverSampler/#1]: Aminian, Ehsan, Rita P. Ribeiro, and João Gama. "Chebyshev approaches for imbalanced data streams regression models." Data Mining and Knowledge Discovery 35.6 (2021): 2389-2466.
