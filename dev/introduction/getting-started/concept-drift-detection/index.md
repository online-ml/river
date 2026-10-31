# Concept drift

In online machine learning, it is assumed that data can change over time. When building machine learning models, we assume data has a probability distribution, which is usually fixed, i.e., stationary. Changes in the data distribution give rise to the phenomenon called **Concept drift**. Such drifts can be either *virtual* or *real*. In virtual drifts, only the distribution of the features, \\(P(X)\\), changes, whereas the relationship between \\(X\\) (features) and the target, \\(y\\), remains unchanged. The joint probability of \\(P(X, y)\\) changes in real concept drifts. Consequently, non-supervised online machine learning problems might face only virtual concept drifts.

Real concept drifts can be further divided in *abrupt* (happen instantly at a given point) or *gradual* (one "concept" changes to another gradually). There are other possible divisions, but they can be fit into abrupt or gradual drifts.

## Examples of concept drift

Concept drifts might happen in the electricity demand across the year, in the stock market, in buying preferences, and in the likelihood of a new movie's success, among others.

Let us consider the movie example: two movies made at different epochs can have similar features such as famous actors/directors, storyline, production budget, marketing campaigns, etc., yet it is not certain that both will be similarly successful. What the target audience *considers* is worth watching (and their money worth spending) is constantly changing, and production companies must adapt accordingly to avoid "box office flops".

Prior to the pandemic, the usage of hand sanitizers and facial masks was not widespread. When the cases of COVID-19 started increasing, there was a lack of such products for the end consumer. Imagine a batch-learning model deciding how much of each product a supermarket should stock during those times. What a mess!

## Impact of drift on learning

Concept drift can have a significant impact on predictive performance if not handled properly. Most batch learning models will fail in the presence of concept drift as they are essentially trained on different data. On the other hand, stream learning methods continuously update themselves and adapt to new concepts. Furthermore, drift-aware methods use change detection methods (a.k.a. drift detectors) to trigger *mitigation mechanisms* if a change in performance is detected.

## Detecting concept drift

Multiple drift detection methods have been proposed. The goal of a drift detector is to signal an alarm in the presence of drift. A good drift detector maximizes the number of true positives while keeping the number of false positives to a minimum. It must also be resource-wise efficient to work in the context of infinite data streams.

For this example, we will generate a synthetic data stream by concatenating 3 distributions of 1000 samples each:

- \\(dist\_a\\): \\(\\mu=0.8\\), \\(\\sigma=0.05\\)
- \\(dist\_b\\): \\(\\mu=0.4\\), \\(\\sigma=0.02\\)
- \\(dist\_c\\): \\(\\mu=0.6\\), \\(\\sigma=0.1\\).

```python
import numpy as np
import altair as alt
import pandas as pd

# Generate data
random_state = np.random.RandomState(seed=42)
dist_a = random_state.normal(0.8, 0.05, 1000)
dist_b = random_state.normal(0.4, 0.02, 1000)
dist_c = random_state.normal(0.6, 0.1, 1000)

stream = np.concatenate((dist_a, dist_b, dist_c))


def plot_data(dist_a, dist_b, dist_c, drifts=None):


    stream_df = pd.DataFrame({
        "Index": list(range(len(stream))),
        "Value": stream
    })

    stream_chart = alt.Chart(stream_df).mark_line().encode(
        x=alt.X("Index", title="Index"),
        y=alt.Y("Value", title="Stream")
    ).properties(
        width=400,
        height=200
    )

    # Add drift lines
    if drifts is not None:
        drift_df = pd.DataFrame({"x": drifts})
        drift_lines = alt.Chart(drift_df).mark_rule(color="red").encode(x="x")
        stream_chart = stream_chart + drift_lines


    hist_df = pd.DataFrame({
        "value": np.concatenate([dist_a, dist_b, dist_c]),
        "distribution": (
            ["dist_a"] * len(dist_a) +
            ["dist_b"] * len(dist_b) +
            ["dist_c"] * len(dist_c)
        )
    })

    hist_chart = alt.Chart(hist_df).mark_bar(opacity=0.5).encode(
        x=alt.X("value:Q", bin=True, title="Value"),
        y=alt.Y("count()", title="Frequency"),
        color="distribution:N"
    ).properties(
        width=200,
        height=200
    )


    final_chart = (stream_chart | hist_chart)

    return final_chart


# Run
plot_data(dist_a, dist_b, dist_c)
```

### Drift detection test

We will use the ADaptive WINdowing (`ADWIN`) drift detection method. Remember that the goal is to indicate that drift has occurred after samples **1000** and **2000** in the synthetic data stream.

```python
from river import drift

drift_detector = drift.ADWIN()
drifts = []

for i, val in enumerate(stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        print(f'Change detected at index {i}')
        drifts.append(i)

plot_data(dist_a, dist_b, dist_c, drifts)
```

```text
Change detected at index 1023
Change detected at index 2047
```

We see that `ADWIN` successfully indicates the presence of drift (red vertical lines) close to the beginning of a new data distribution.

---

We conclude this example with some remarks regarding concept drift detectors and their usage:

- In practice, drift detectors provide stream learning methods with robustness against concept drift. Drift detectors monitor the model usually through a performance metric.
- Drift detectors work on univariate data. This is why they are used to monitor a model's performance and not the data itself. Remember that concept drift is defined as a change in the relationship between data and the target to learn (in supervised learning).
- Drift detectors define their expectations regarding input data. It is important to know these expectations to feed a given drift detector with the correct data.
