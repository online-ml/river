# Binary classification

Classification is about predicting an outcome from a fixed list of classes. The prediction is a probability distribution that assigns a probability to each possible outcome.

A labeled classification sample is made up of a bunch of features and a class. The class is a boolean in the case of binary classification. We'll use the phishing dataset as an example.


```python
from river import datasets

dataset = datasets.Phishing()
dataset
```




    Phishing websites.
    
    This dataset contains features from web pages that are classified as phishing or not.
    
        Name  Phishing                                                        
        Task  Binary classification                                           
     Samples  1,250                                                           
    Features  9                                                               
      Sparse  False                                                           
        Path  /Users/max.halford/projects/river/river/datasets/phishing.csv.gz



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




    {'empty_server_form_handler': 0.0,
     'popup_window': 0.0,
     'https': 0.0,
     'request_from_other_domain': 0.0,
     'anchor_from_other_domain': 0.0,
     'is_popular': 0.5,
     'long_url': 1.0,
     'age_of_domain': 1,
     'ip_in_url': 1}




```python
y
```




    True



The goal of a classification is to train a model which is able to predict `y` from `x`. We'll try to do this with a logistic regression.


```python
from river import linear_model

model = linear_model.LogisticRegression()
model.predict_proba_one(x)
```




    {False: 0.5, True: 0.5}



The model hasn't been trained on any data, and therefore outputs a default probability of 50% for each class.

The model can be trained on the sample, which will update the model's state.


```python
model = model.learn_one(x, y)
```

If we try to make a prediction on the same sample, we can see that the probabilities are different, because the model has learned something.


```python
model.predict_proba_one(x)
```




    {False: 0.5068745667645342, True: 0.4931254332354657}



Note that there is also a `predict_one` if you're only interested in the most likely class rather than the probability distribution.


```python
model.predict_one(x)
```




    False



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




    ROCAUC: 89.36%



This is a common way to evaluate an online model. In fact, there is a dedicated `evaluate.progressive_val_score` function that does this for you.


```python
from river import evaluate

model = linear_model.LogisticRegression()
metric = metrics.ROCAUC()

evaluate.progressive_val_score(dataset, model, metric)
```




    ROCAUC: 89.36%



A common way to improve the performance of a logistic regression is to scale the data. This can be done by using a `preprocessing.StandardScaler`. In particular, we can define a pipeline to organise our model into a sequence of steps:


```python
from river import compose
from river import preprocessing

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)

model
```




<div><div class="component pipeline"><details class="component estimator"><summary><pre class="estimator-name">StandardScaler</pre></summary><code class="estimator-params">
{'counts': Counter(),
 'means': defaultdict(&lt;class 'float'&gt;, {}),
 'vars': defaultdict(&lt;class 'float'&gt;, {}),
 'with_std': True}

</code></details><details class="component estimator"><summary><pre class="estimator-name">LogisticRegression</pre></summary><code class="estimator-params">
{'_weights': {},
 '_y_name': None,
 'clip_gradient': 1000000000000.0,
 'initializer': Zeros (),
 'intercept': 0.0,
 'intercept_init': 0.0,
 'intercept_lr': Constant({'learning_rate': 0.01}),
 'l1': 0.0,
 'l2': 0.0,
 'loss': Log({'weight_pos': 1.0, 'weight_neg': 1.0}),
 'optimizer': SGD({'lr': Constant({'learning_rate': 0.01}), 'n_iterations': 0})}

</code></details></div><style scoped>
.estimator {
    padding: 1em;
    border-style: solid;
    background: white;
}

.pipeline {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: linear-gradient(#000, #000) no-repeat center / 3px 100%;
}

.union {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white
}

.wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white;
}

.wrapper > .estimator {
    margin-top: 1em;
}

/* Vertical spacing between steps */

.component + .component {
    margin-top: 2em;
}

.union > .estimator {
    margin-top: 0;
}

.union > .pipeline {
    margin-top: 0;
}

/* Spacing within a union of estimators */

.union > .component + .component {
    margin-left: 1em;
}

/* Typography */

.estimator-params {
    display: block;
    white-space: pre-wrap;
    font-size: 120%;
    margin-bottom: -1em;
}

.estimator > code,
.wrapper > details > code {
    background-color: white !important;
}

.estimator-name {
    display: inline;
    margin: 0;
    font-size: 130%;
}

/* Toggle */

summary {
    display: flex;
    align-items:center;
    cursor: pointer;
}

summary > div {
    width: 100%;
}
</style></div>




```python
metric = metrics.ROCAUC()
evaluate.progressive_val_score(dataset, model, metric)
```




    ROCAUC: 95.04%



That concludes the getting started introduction to classification! You can now move on to the [next steps](/introduction/next-steps).
