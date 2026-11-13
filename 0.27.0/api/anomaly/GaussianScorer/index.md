# GaussianScorer

Univariate Gaussian anomaly detector.

This is a supervised anomaly detector. It fits a Gaussian distribution to the target values. The anomaly score is then computed as so:

\\\[score = 2 \\mid CDF(y) - 0.5 \\mid\\\]

This makes it so that the anomaly score is between 0 and 1.

## Parameters

- **window\_size**

  *Default* → `None`

  Set this to fit the Gaussian distribution over a window of recent values.
- **grace\_period**

  *Default* → `100`

  Number of samples before which a 0 is always returned. This is handy because the Gaussian distribution needs time to stabilize, and will likely produce overly high anomaly score for the first samples.

## Examples

```python
import random
from river import anomaly

rng = random.Random(42)
detector = anomaly.GaussianScorer()

for y in (rng.gauss(0, 1) for _ in range(100)):
    detector.learn_one(None, y)

detector.score_one(None, -3)
```

```text
0.999477...
```

```python
detector.score_one(None, 3)
```

```text
0.999153...
```

```python
detector.score_one(None, 0)
```

```text
0.052665...
```

```python
detector.score_one(None, 0.5)
```

```text
0.383717...
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.Target`

> [!NOTE]
>
> **score\_one**
>
> Return an outlier score.
>
> A high score is indicative of an anomaly. A low score corresponds a normal observation.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.Target`
>
> **Returns**
>
> `float`: An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
