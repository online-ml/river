# Regression

Regression is about predicting a numeric output for a given sample. A labeled regression sample is made up of a bunch of features and a number. The number is usually continuous, but it may also be discrete. We'll use the Trump approval rating dataset as an example.

```python
from river import datasets

dataset = datasets.TrumpApproval()
dataset
```

```
```

```text
Donald Trump approval ratings.

This dataset was obtained by reshaping the data used by FiveThirtyEight for analyzing Donald
Trump's approval ratings. It contains [1;36m5[0m features, which are approval ratings collected by
[1;36m5[0m polling agencies. The target is the approval rating from FiveThirtyEight's model. The goal of
this task is to see if we can reproduce FiveThirtyEight's model.

References
----------
[1m[[0m^[1;36m1[0m[1m][0m: [1m[[0mTrump Approval Ratings[1m][0m[1m([0m[4;94mhttps://projects.fivethirtyeight.com/trump-approval-ratings/[0m[4;94m)[0m

    Name  TrumpApproval                                                     
    Task  Regression                                                        
 Samples  [1;36m1[0m,[1;36m001[0m                                                             
Features  [1;36m6[0m                                                                 
  Sparse  [3;91mFalse[0m                                                             
    Path  [35m/home/runner/work/river/river/river/datasets/[0m[95mtrump_approval.csv.gz[0m
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

```
```

```text
[1m{[0m
    [32m'ordinal_date'[0m: [1;36m736389[0m,
    [32m'gallup'[0m: [1;36m43.843213[0m,
    [32m'ipsos'[0m: [1;36m46.19925042857143[0m,
    [32m'morning_consult'[0m: [1;36m48.318749[0m,
    [32m'rasmussen'[0m: [1;36m44.104692[0m,
    [32m'you_gov'[0m: [1;36m43.636914000000004[0m
[1m}[0m
```

A regression model's goal is to learn to predict a numeric target `y` from a bunch of features `x`. We'll attempt to do this with a nearest neighbors model.

```python
from river import neighbors

model = neighbors.KNNRegressor()
model.predict_one(x)
```

```
```

```text
[1;36m0.0[0m
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

```
```

```text
[1;36m43.75505[0m
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

```
```

```text
MAE: [1;36m0.310353[0m
```

This is a common way to evaluate an online model. In fact, there is a dedicated [`evaluate.progressive_val_score`](<https://riverml.xyz/0.27.0/api/evaluate/progressive-val-score>) function that does this for you.

```python
from river import evaluate

model = neighbors.KNNRegressor()
metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric)
```

```
```

```text
MAE: [1;36m0.310353[0m
```
