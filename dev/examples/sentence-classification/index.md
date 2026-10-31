# Sentence classification

In this tutorial we will try to predict whether an SMS is a spam or not. To train our model, we will use the `SMSSpam` dataset. This dataset is unbalanced, there is only 13.4% spam. Let's look at the data:

```python
from river import datasets

datasets.SMSSpam()
```

```text
SMS Spam Collection dataset.

The data contains 5,574 items and 1 feature (i.e. SMS body). Spam messages represent
13.4% of the dataset. The goal is to predict whether an SMS is a spam or not.

      Name  SMSSpam                                                                              
      Task  Binary classification                                                                
   Samples  5,574                                                                                
  Features  1                                                                                    
    Sparse  False                                                                                
      Path  /Users/max/river_data/SMSSpam/SMSSpamCollection                                      
       URL  https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
      Size  466.71 KB                                                                            
Downloaded  True
```

```python
from pprint import pprint

X_y = datasets.SMSSpam()

for x, y in X_y:
    pprint(x)
    print(f'Spam: {y}')
    break
```

```text
{'body': 'Go until jurong point, crazy.. Available only in bugis n great world '
         'la e buffet... Cine there got amore wat...\n'}
Spam: False
```

Let's start by building a simple model like a Naive Bayes classifier. We will first preprocess the sentences with a TF-IDF transform that our model can consume. Then, we will measure the accuracy of our model with the AUC metric. This is the right metric to use when the classes are not balanced. In addition, the Naive Bayes models can perform very well on unbalanced datasets and can be used for both binary and multi-class classification problems.

```python
from river import feature_extraction
from river import naive_bayes
from river import metrics

X_y = datasets.SMSSpam()

model = (
    feature_extraction.TFIDF(on='body') | 
    naive_bayes.BernoulliNB(alpha=0)
)

metric = metrics.ROCAUC()
cm = metrics.ConfusionMatrix()

for x, y in X_y:

    y_pred = model.predict_one(x)

    if y_pred is not None:
        metric.update(y_pred=y_pred, y_true=y)
        cm.update(y_pred=y_pred, y_true=y)

    model.learn_one(x, y)

metric
```

```text
ROCAUC: 93.00%
```

The confusion matrix:

```python
cm
```

```text
False   True  
False   4,809     17  
 True     102    645
```

The results are quite good with this first model.

Since we are working with an imbalanced dataset, we can use the `imblearn` module to rebalance the classes of our dataset. For more information about the `imblearn` module, you can find a dedicated tutorial [here](<https://riverml.xyz/dev/imbalanced-learning>).

```python
from river import imblearn

X_y = datasets.SMSSpam()

model = (
    feature_extraction.TFIDF(on='body') | 
    imblearn.RandomUnderSampler(
        classifier=naive_bayes.BernoulliNB(alpha=0),
        desired_dist={0: .5, 1: .5},
        seed=42
    )
)

metric = metrics.ROCAUC()
cm = metrics.ConfusionMatrix()

for x, y in X_y:

    y_pred = model.predict_one(x)

    if y_pred is not None:
        metric.update(y_pred=y_pred, y_true=y)
        cm.update(y_pred=y_pred, y_true=y)

    model.learn_one(x, y)

metric
```

```text
ROCAUC: 94.61%
```

The `imblearn` module improved our results. Not bad! We can visualize the pipeline to understand how the data is processed.

The confusion matrix:

```python
cm
```

```text
False   True  
False   4,570    255  
 True      41    706
```

```python
model
```

**```
TFIDF
```**

`TFIDF (   normalize=True   on="body"   strip_accents=True   lowercase=True   preprocessor=None   tokenizer=None   ngram_range=(1, 1) ) `

**```
RandomUnderSampler
```**

`RandomUnderSampler (   classifier=BernoulliNB (     alpha=0     true_threshold=0.   )   desired_dist={0: 0.5, 1: 0.5}   seed=42 ) `

**```
BernoulliNB
```**

`BernoulliNB (   alpha=0   true_threshold=0. ) `

Now let's try to use logistic regression to classify messages. We will use different tips to make my model perform better. As in the previous example, we rebalance the classes of our dataset. The logistics regression will be fed from a TF-IDF.

```python
from river import linear_model
from river import optim
from river import preprocessing

X_y = datasets.SMSSpam()

model = (
    feature_extraction.TFIDF(on='body') | 
    preprocessing.Normalizer() | 
    imblearn.RandomUnderSampler(
        classifier=linear_model.LogisticRegression(
            optimizer=optim.SGD(.9), 
            loss=optim.losses.Log()
        ),
        desired_dist={0: .5, 1: .5},
        seed=42
    )
)

metric = metrics.ROCAUC()
cm = metrics.ConfusionMatrix()

for x, y in X_y:

    y_pred = model.predict_one(x)

    metric.update(y_pred=y_pred, y_true=y)
    cm.update(y_pred=y_pred, y_true=y)

    model.learn_one(x, y)

metric
```

```text
ROCAUC: 93.80%
```

The confusion matrix:

```python
cm
```

```text
False   True  
False   4,584    243  
 True      55    692
```

```python
model
```

**```
TFIDF
```**

`TFIDF (   normalize=True   on="body"   strip_accents=True   lowercase=True   preprocessor=None   tokenizer=None   ngram_range=(1, 1) ) `

**```
Normalizer
```**

`Normalizer (   order=2 ) `

**```
RandomUnderSampler
```**

`RandomUnderSampler (   classifier=LogisticRegression (     optimizer=SGD (       lr=Constant (         learning_rate=0.9       )     )     loss=Log (       weight_pos=1.       weight_neg=1.     )     l2=0.     l1=0.     intercept_init=0.     intercept_lr=Constant (       learning_rate=0.01     )     clip_gradient=1e+12     initializer=Zeros ()   )   desired_dist={0: 0.5, 1: 0.5}   seed=42 ) `

**```
LogisticRegression
```**

`LogisticRegression (   optimizer=SGD (     lr=Constant (       learning_rate=0.9     )   )   loss=Log (     weight_pos=1.     weight_neg=1.   )   l2=0.   l1=0.   intercept_init=0.   intercept_lr=Constant (     learning_rate=0.01   )   clip_gradient=1e+12   initializer=Zeros () ) `

The results of the logistic regression are quite good but still inferior to the naive Bayes model.
