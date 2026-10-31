# Binary classification

Classification is about predicting an outcome from a fixed list of classes. The prediction is a probability distribution that assigns a probability to each possible outcome.

A labeled classification sample is made up of a bunch of features and a class. The class is a boolean in the case of binary classification. We'll use the phishing dataset as an example.

```python
from river import datasets

dataset = datasets.Phishing()
dataset
```

```text
Phishing websites.

This dataset contains features from web pages that are classified as phishing or not.

References
----------
[^1]: [UCI page](http://archive.ics.uci.edu/ml/datasets/Website+Phishing)

    Name  Phishing                                                
    Task  Binary classification                                   
 Samples  1,250                                                   
Features  9                                                       
  Sparse  False                                                   
    Path  /Users/max/projects/river/river/datasets/phishing.csv.gz
```

This dataset is a streaming dataset which can be looped over.

```python
for x, y in dataset:
    pass
```

Let's take a look at the first sample.

```python
x, y = next(iter(dataset))
x
```

```text
{'empty_server_form_handler': 0.0,
 'popup_window': 0.0,
 'https': 0.0,
 'request_from_other_domain': 0.0,
 'anchor_from_other_domain': 0.0,
 'is_popular': 0.5,
 'long_url': 1.0,
 'age_of_domain': 1,
 'ip_in_url': 1}
```

```python
y
```

```text
True
```

A binary classifier's goal is to learn to predict a binary target `y` from some given features `x`. We'll try to do this with a logistic regression.

```python
from river import linear_model

model = linear_model.LogisticRegression()
model.predict_proba_one(x)
```

```text
{False: 0.5, True: 0.5}
```

The model hasn't been trained on any data, and therefore outputs a default probability of 50% for each class.

The model can be trained on the sample, which will update the model's state.

```python
model.learn_one(x, y)
```

If we try to make a prediction on the same sample, we can see that the probabilities are different, because the model has learned something.

```python
model.predict_proba_one(x)
```

```text
{False: 0.494687699901455, True: 0.505312300098545}
```

Note that there is also a `predict_one` if you're only interested in the most likely class rather than the probability distribution.

```python
model.predict_one(x)
```

```text
True
```

Typically, an online model makes a prediction, and then learns once the ground truth reveals itself. The prediction and the ground truth can be compared to measure the model's correctness. If you have a dataset available, you can loop over it, make a prediction, update the model, and compare the model's output with the ground truth. This is called progressive validation.

```python
from river import metrics

model = linear_model.LogisticRegression()

metric = metrics.ROCAUC()

for x, y in dataset:
    y_pred = model.predict_proba_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)

metric
```

```text
ROCAUC: 89.36%
```

This is a common way to evaluate an online model. In fact, there is a dedicated [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>) function that does this for you.

```python
from river import evaluate

model = linear_model.LogisticRegression()
metric = metrics.ROCAUC()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
ROCAUC: 89.36%
```

A common way to improve the performance of a logistic regression is to scale the data. This can be done by using a [`preprocessing.StandardScaler`](<https://riverml.xyz/dev/api/preprocessing/StandardScaler>). In particular, we can define a pipeline to organise our model into a sequence of steps:

```python
from river import compose
from river import preprocessing

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)

model
```

**```
StandardScaler
```**

`StandardScaler (   with_std=True ) `

**```
LogisticRegression
```**

`LogisticRegression (   optimizer=SGD (     lr=Constant (       learning_rate=0.01     )   )   loss=Log (     weight_pos=1.     weight_neg=1.   )   l2=0.   l1=0.   intercept_init=0.   intercept_lr=Constant (     learning_rate=0.01   )   clip_gradient=1e+12   initializer=Zeros () ) `

```python
metric = metrics.ROCAUC()
evaluate.progressive_val_score(dataset, model, metric)
```

```text
ROCAUC: 95.07%
```
