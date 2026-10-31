# ChebyshevUnderSampler

Under-sampling for imbalanced regression using Chebyshev's inequality.

Chebyshev's inequality can be used to define the probability of target observations being frequent values (w.r.t. the distribution mean).

Let \\(Y\\) be a random variable with finite expected value \\(\\overline{y}\\) and non-zero variance \\(\\sigma^2\\). For any real number \\(t \> 0\\), the Chebyshev's inequality states that, for a wide class of unimodal probability distributions: \\(Pr(|y-\\overline{y}| \\ge t\\sigma) \\le \\dfrac{1}{t^2}\\).

Taking \\(t=\\dfrac{|y-\\overline{y}|}{\\sigma}\\), and assuming \\(t \> 1\\), the Chebyshev’s inequality for an observation \\(y\\) becomes: \\(P(|y - \\overline{y}|=t) = \\dfrac{\\sigma^2}{|y-\\overline{y}|}\\). The reciprocal of this probability is used for under-sampling[^/dev/api/imblearn/ChebyshevUnderSampler/#1] the most frequent cases. Extreme valued or rare cases have higher probabilities of selection, whereas the most frequent cases are likely to be discarded. Still, frequent cases have a small chance of being selected (controlled via the `sp` parameter) in case few rare instances were observed.

## Parameters

- **regressor**

  *Type* → `base.Regressor`

  The regression model that will receive the biased sample.
- **sp**

  *Type* → `float`

  *Default* → `0.15`

  Second chance probability. Even if an example is not initially selected for training, it still has a small chance of being selected in case the number of rare case observed so far is small.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed to support reproducibility.

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
    imblearn.ChebyshevUnderSampler(
        regressor=rules.AMRules(
            n_min=50, delta=0.01,
        ),
        seed=42
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
[500] MAE: 1.787162
[1,000] MAE: 1.515711
[1,001] MAE: 1.515236
MAE: 1.515236
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

[^/dev/api/imblearn/ChebyshevUnderSampler/#1]: Aminian, Ehsan, Rita P. Ribeiro, and João Gama. "Chebyshev approaches for imbalanced data streams regression models." Data Mining and Knowledge Discovery 35.6 (2021): 2389-2466.
