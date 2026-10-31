# MultivariateGaussian

Multivariate normal distribution with parameters mu and var.

## Parameters

- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **mode**

  The most likely value in the distribution.
- **mu**

  The mean value of the distribution.
- **n\_samples**

  The number of observed samples.
- **sigma**

  The standard deviation of the distribution.
- **var**

  The variance of the distribution.

## Examples

```python
import numpy as np
import pandas as pd
from river import proba

np.random.seed(42)
X = pd.DataFrame(
    np.random.random((8, 3)),
    columns=["red", "green", "blue"]
)
X
```

```text
        red     green      blue
0  0.374540  0.950714  0.731994
1  0.598658  0.156019  0.155995
2  0.058084  0.866176  0.601115
3  0.708073  0.020584  0.969910
4  0.832443  0.212339  0.181825
5  0.183405  0.304242  0.524756
6  0.431945  0.291229  0.611853
7  0.139494  0.292145  0.366362
```

```python
p = proba.MultivariateGaussian(seed=42)
p.n_samples
```

```text
0.0
```

```python
for x in X.to_dict(orient="records"):
    p.update(x)
p.var
```

```text
           blue     green       red
blue   0.076119  0.020292 -0.010128
green  0.020292  0.112931 -0.053268
red   -0.010128 -0.053268  0.078961
```

Retrieving current state in nice format is simple

```python
p
```

```text
𝒩(
    μ=(0.518, 0.387, 0.416),
    σ^2=(
        [0.076 0.020 -0.010]
        [0.020 0.113 -0.053]
        [-0.010 -0.053 0.079]
    )
)
```

To retrieve number of samples and mode:

```python
p.n_samples
```

```text
8.0
```

```python
p.mode
```

```text
{'blue': 0.5179..., 'green': 0.3866..., 'red': 0.4158...}
```

To retrieve the PDF and CDF:

```python
p(x)
```

```text
0.97967...
```

```python
p.cdf(x)
```

```text
0.005096...
```

To sample data from distribution:

```python
p.sample()
```

```text
{'blue': -0.179..., 'green': -0.051..., 'red': 0.376...}
```

MultivariateGaussian works with [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>):

```python
from river import utils

p = utils.Rolling(MultivariateGaussian, window_size=5)
for x in X.to_dict(orient="records"):
    p.update(x)
p.var
```

```text
           blue     green       red
blue   0.087062 -0.022873  0.007765
green -0.022873  0.014279 -0.025181
red    0.007765 -0.025181  0.095066
```

MultivariateGaussian works with [`utils.TimeRolling`](<https://riverml.xyz/dev/api/utils/TimeRolling>):

```python
from datetime import datetime as dt, timedelta as td
X.index = [dt(2023, 3, 28, 0, 0, 0) + td(seconds=x) for x in range(8)]
p = utils.TimeRolling(MultivariateGaussian, period=td(seconds=5))
for t, x in X.iterrows():
    p.update(x.to_dict(), t=t)
p.var
```

```text
           blue     green       red
blue   0.087062 -0.022873  0.007765
green -0.022873  0.014279 -0.025181
red    0.007765 -0.025181  0.095066
```

Variance on diagonal is consistent with [`proba.Gaussian`](<https://riverml.xyz/dev/api/proba/Gaussian>).

```python
multi = proba.MultivariateGaussian()
single = proba.Gaussian()
for x in X.to_dict(orient='records'):
    multi.update(x)
    single.update(x['blue'])
multi.mu['blue'] == single.mu
```

```text
True
```

```python
multi.sigma['blue']['blue'] == single.sigma
```

```text
np.True_
```

## Methods

> [!NOTE]
>
> ****call****
>
> PDF(x) method.
>
> **Parameters**
>
> - **x** — `dict[str, float]`

> [!NOTE]
>
> **cdf**
>
> Cumulative density function, i.e. P(X \<= x).
>
> **Parameters**
>
> - **x** — `dict[str, float]`

> [!NOTE]
>
> **revert**
>
> Reverts the parameters of the distribution for a given observation.
>
> **Parameters**
>
> - **x** — `dict[str, float]`
> - **w** — `float` — defaults to `1.0`

> [!NOTE]
>
> **sample**
>
> Sample a random value from the distribution.

> [!NOTE]
>
> **update**
>
> Updates the parameters of the distribution given a new observation.
>
> **Parameters**
>
> - **x** — `dict[str, float]`
> - **w** — `float` — defaults to `1.0`
