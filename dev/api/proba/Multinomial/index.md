# Multinomial

Multinomial distribution for categorical data.

## Parameters

- **events**

  *Type* → `dict[typing.Any, int] | list[typing.Any] | None`

  *Default* → `None`

  An optional list of events that already occurred.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Attributes

- **mode**

  The most likely value in the distribution.
- **n\_samples**

  The number of observed samples.

## Examples

```python
from river import proba

p = proba.Multinomial(['green'] * 3)
p.update('red')
p('red')
```

```text
0.25
```

```python
p.update('red')
p.update('red')
p('green')
```

```text
0.5
```

```python
p.revert('red')
p.revert('red')
p('red')
```

```text
0.25
```

You can wrap this with a [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>) to measure a distribution over a window:

```python
from river import utils

X = ['red', 'green', 'green', 'blue', 'blue']

dist = utils.Rolling(
    proba.Multinomial,
    window_size=3
)

for x in X:
    dist.update(x)
    print(dist)
    print()
```

```text
P(red) = 1.000
<BLANKLINE>
P(red) = 0.500
P(green) = 0.500
<BLANKLINE>
P(green) = 0.667
P(red) = 0.333
<BLANKLINE>
P(green) = 0.667
P(blue) = 0.333
P(red) = 0.000
<BLANKLINE>
P(blue) = 0.667
P(green) = 0.333
P(red) = 0.000
<BLANKLINE>
```

You can wrap this with a [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>) to measure a distribution over a window of time:

```python
import datetime as dt

X = ['red', 'green', 'green', 'blue']
days = [1, 2, 3, 4]

dist = utils.TimeRolling(
    proba.Multinomial,
    period=dt.timedelta(days=2)
)

for x, day in zip(X, days):
    dist.update(x, t=dt.datetime(2019, 1, day))
    print(dist)
    print()
```

```text
P(red) = 1.000
<BLANKLINE>
P(red) = 0.500
P(green) = 0.500
<BLANKLINE>
P(green) = 1.000
P(red) = 0.000
<BLANKLINE>
P(green) = 0.500
P(blue) = 0.500
P(red) = 0.000
<BLANKLINE>
```

## Methods

> [!NOTE]
>
> ****call****
>
> Probability mass/density function.
>
> **Parameters**
>
> - **x** — `typing.Hashable`

> [!NOTE]
>
> **revert**
>
> Reverts the parameters of the distribution for a given observation.
>
> **Parameters**
>
> - **x** — `typing.Hashable`

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
> - **x** — `typing.Hashable`
