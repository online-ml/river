# Working with imbalanced data

In machine learning it is quite usual to have to deal with imbalanced dataset. This is particularly true in online learning for tasks such as fraud detection and spam classification. In these two cases, which are binary classification problems, there are usually many more 0s than 1s, which generally hinders the performance of the classifiers we thrown at them.

As an example we'll use the credit card dataset available in River. We'll first use a `collections.Counter` to count the number of 0s and 1s in order to get an idea of the class balance.

```python
import collections
from river import datasets

X_y = datasets.CreditCard()

counts = collections.Counter(y for _, y in X_y)

for c, count in counts.items():
    print(f'{c}: {count} ({count / sum(counts.values()):.5%})')
```

```text
0: 284315 (99.82725%)
1: 492 (0.17275%)
```

## Baseline

The dataset is quite unbalanced. For each 1 there are about 578 0s. Let's now train a logistic regression with default parameters and see how well it does. We'll measure the ROC AUC score.

```python
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing


X_y = datasets.CreditCard()

model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression()
)

metric = metrics.ROCAUC()

evaluate.progressive_val_score(X_y, model, metric)
```

```text
ROCAUC: 89.11%
```

## Importance weighting

The performance is already quite acceptable, but as we will now see we can do even better. The first thing we can do is to add weight to the 1s by using the `weight_pos` argument of the `Log` loss function.

```python
from river import optim

model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(
        loss=optim.losses.Log(weight_pos=5)
    )
)

metric = metrics.ROCAUC()

evaluate.progressive_val_score(X_y, model, metric)
```

```text
ROCAUC: 91.43%
```

## Focal loss

The deep learning for object detection community has produced a special loss function for imbalanced learning called [focal loss](<https://arxiv.org/pdf/1708.02002.pdf>). We are doing binary classification, so we can plug the binary version of focal loss into our logistic regression and see how well it fairs.

```python
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression(loss=optim.losses.BinaryFocalLoss(2, 1))
)

metric = metrics.ROCAUC()

evaluate.progressive_val_score(X_y, model, metric)
```

```text
ROCAUC: 91.31%
```

## Under-sampling the majority class

Adding importance weights only works with gradient-based models (which includes neural networks). A more generic, and potentially more effective approach, is to use undersamplig and oversampling. As an example, we'll under-sample the stream so that our logistic regression encounter 20% of 1s and 80% of 0s. Under-sampling has the additional benefit of requiring less training steps, and thus reduces the total training time.

```python
from river import imblearn

model = (
    preprocessing.StandardScaler() |
    imblearn.RandomUnderSampler(
        classifier=linear_model.LogisticRegression(),
        desired_dist={0: .8, 1: .2},
        seed=42
    )
)

metric = metrics.ROCAUC()

evaluate.progressive_val_score(X_y, model, metric)
```

```text
ROCAUC: 94.75%
```

The `RandomUnderSampler` class is a wrapper for classifiers. This is represented by a rectangle around the logistic regression bubble when we visualize the model.

```python
model
```

**```
StandardScaler
```**

`StandardScaler (   with_std=True ) `

**```
RandomUnderSampler
```**

`RandomUnderSampler (   classifier=LogisticRegression (     optimizer=SGD (       lr=Constant (         learning_rate=0.01       )     )     loss=Log (       weight_pos=1.       weight_neg=1.     )     l2=0.     l1=0.     intercept_init=0.     intercept_lr=Constant (       learning_rate=0.01     )     clip_gradient=1e+12     initializer=Zeros ()   )   desired_dist={0: 0.8, 1: 0.2}   seed=42 ) `

**```
LogisticRegression
```**

`LogisticRegression (   optimizer=SGD (     lr=Constant (       learning_rate=0.01     )   )   loss=Log (     weight_pos=1.     weight_neg=1.   )   l2=0.   l1=0.   intercept_init=0.   intercept_lr=Constant (     learning_rate=0.01   )   clip_gradient=1e+12   initializer=Zeros () ) `

## Over-sampling the minority class

We can also attain the same class distribution by over-sampling the minority class. This will come at cost of having to train with more samples.

```python
model = (
    preprocessing.StandardScaler() |
    imblearn.RandomOverSampler(
        classifier=linear_model.LogisticRegression(),
        desired_dist={0: .8, 1: .2},
        seed=42
    )
)

metric = metrics.ROCAUC()

evaluate.progressive_val_score(X_y, model, metric)
```

```text
ROCAUC: 91.71%
```

## Sampling with a desired sample size

The downside of both `RandomUnderSampler` and `RandomOverSampler` is that you don't have any control on the amount of data the classifier trains on. The number of samples is adjusted so that the target distribution can be attained, either by under-sampling or over-sampling. However, you can do both at the same time and choose how much data the classifier will see. To do so, we can use the `RandomSampler` class. In addition to the desired class distribution, we can specify how much data to train on. The samples will both be under-sampled and over-sampled in order to fit your constraints. This is powerful because it allows you to control both the class distribution and the size of the training data (and thus the training time). In the following example we'll set it so that the model will train with 1 percent of the data.

```python
model = (
    preprocessing.StandardScaler() |
    imblearn.RandomSampler(
        classifier=linear_model.LogisticRegression(),
        desired_dist={0: .8, 1: .2},
        sampling_rate=.01,
        seed=42
    )
)

metric = metrics.ROCAUC()

evaluate.progressive_val_score(X_y, model, metric)
```

```text
ROCAUC: 94.71%
```

## Hybrid approach

As you might have guessed by now, nothing is stopping you from mixing imbalanced learning methods together. As an example, let's combine `sampling.RandomUnderSampler` and the `weight_pos` parameter from the [`optim.losses.Log`](<https://riverml.xyz/dev/api/optim/losses/Log>) loss function.

```python
model = (
    preprocessing.StandardScaler() |
    imblearn.RandomUnderSampler(
        classifier=linear_model.LogisticRegression(
            loss=optim.losses.Log(weight_pos=5)
        ),
        desired_dist={0: .8, 1: .2},
        seed=42
    )
)

metric = metrics.ROCAUC()

evaluate.progressive_val_score(X_y, model, metric)
```

```text
ROCAUC: 96.52%
```
