# Regression

Regression is about predicting a numeric output for a given sample. A labeled regression sample is made up of a bunch of features and a number. The number is usually continuous, but it may also be discrete. We'll use the Trump approval rating dataset as an example.

```python
from river import datasets

dataset = datasets.TrumpApproval()
dataset
```

```text
Donald Trump approval ratings.

This dataset was obtained by reshaping the data used by FiveThirtyEight for analyzing Donald
Trump's approval ratings. It contains 5 features, which are approval ratings collected by
5 polling agencies. The target is the approval rating from FiveThirtyEight's model. The goal of
this task is to see if we can reproduce FiveThirtyEight's model.

References
----------
[^1]: [Trump Approval Ratings](https://projects.fivethirtyeight.com/trump-approval-ratings/)

    Name  TrumpApproval                                                 
    Task  Regression                                                    
 Samples  1,001                                                         
Features  6                                                             
  Sparse  False                                                         
    Path  /Users/max/projects/river/river/datasets/trump_approval.csv.gz
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
{'ordinal_date': 736389,
 'gallup': 43.843213,
 'ipsos': 46.19925042857143,
 'morning_consult': 48.318749,
 'rasmussen': 44.104692,
 'you_gov': 43.636914000000004}
```

A regression model's goal is to learn to predict a numeric target `y` from a bunch of features `x`. We'll attempt to do this with a nearest neighbors model.

```python
from river import neighbors

model = neighbors.KNNRegressor()
model.predict_one(x)
```

```text
0.0
```

The model hasn't been trained on any data, and therefore outputs a default value of 0.

The model can be trained on the sample, which will update the model's state.

```python
model.learn_one(x, y)
```

If we try to make a prediction on the same sample, we can see that the output is different, because the model has learned something.

```python
model.predict_one(x)
```

```text
43.75505
```

Typically, an online model makes a prediction, and then learns once the ground truth reveals itself. The prediction and the ground truth can be compared to measure the model's correctness. If you have a dataset available, you can loop over it, make a prediction, update the model, and compare the model's output with the ground truth. This is called progressive validation.

```python
from river import metrics

model = neighbors.KNNRegressor()

metric = metrics.MAE()

for x, y in dataset:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)

metric
```

```text
MAE: 0.310353
```

This is a common way to evaluate an online model. In fact, there is a dedicated [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>) function that does this for you.

```python
from river import evaluate

model = neighbors.KNNRegressor()
metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MAE: 0.310353
```
