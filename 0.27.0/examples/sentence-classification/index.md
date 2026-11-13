# Sentence classification

In this tutorial we will try to predict whether an SMS is a spam or not. To train our model, we will use the `SMSSpam` dataset. This dataset is unbalanced, there is only 13.4% spam. Let's look at the data:

```python
from river import datasets

datasets.SMSSpam()
```

```
```

```text
SMS Spam Collection dataset.

The data contains [1;36m5[0m,[1;36m574[0m items and [1;36m1[0m feature [1m([0mi.e. SMS body[1m)[0m. Spam messages represent
[1;36m13.4[0m% of the dataset. The goal is to predict whether an SMS is a spam or not.

References
----------
[1m[[0m^[1;36m1[0m[1m][0m: [1m[[0mAlmeida, T.A., Hidalgo, J.M.G. and Yamakami, A., [1;36m2011[0m, September. Contributions to the study of SMS spam filtering: new collection and results. In Proceedings of the 11th ACM symposium on Document engineering [1m([0mpp. [1;36m259[0m-[1;36m262[0m[1m)[0m.[1m][0m[1m([0m[4;94mhttp://www.dt.fee.unicamp.br/~tiago/smsspamcollection/doceng11.pdf[0m[4;94m)[0m

      Name  SMSSpam                                                                              
      Task  Binary classification                                                                
   Samples  [1;36m5[0m,[1;36m574[0m                                                                                
  Features  [1;36m1[0m                                                                                    
    Sparse  [3;91mFalse[0m                                                                                
      Path  [35m/home/runner/river_data/SMSSpam/[0m[95mSMSSpamCollection[0m                                    
       URL  [4;94mhttps://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip[0m
      Size  [1;36m466.71[0m KiB                                                                           
Downloaded  [3;92mTrue[0m
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

```
```

```text
ROCAUC: [1;36m93.00[0m%
```

The confusion matrix:

```python
cm
```

```
```

```text
[3;91mFalse[0m   [3;92mTrue[0m  
[3;91mFalse[0m   [1;36m4[0m,[1;36m809[0m     [1;36m17[0m  
 [3;92mTrue[0m     [1;36m102[0m    [1;36m645[0m
```

The results are quite good with this first model.

Since we are working with an imbalanced dataset, we can use the `imblearn` module to rebalance the classes of our dataset. For more information about the `imblearn` module, you can find a dedicated tutorial [here](<https://riverml.xyz/0.27.0/imbalanced-learning>).

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

```
```

```text
ROCAUC: [1;36m94.61[0m%
```

The `imblearn` module improved our results. Not bad! We can visualize the pipeline to understand how the data is processed.

The confusion matrix:

```python
cm
```

```
```

```text
[3;91mFalse[0m   [3;92mTrue[0m  
[3;91mFalse[0m   [1;36m4[0m,[1;36m570[0m    [1;36m255[0m  
 [3;92mTrue[0m      [1;36m41[0m    [1;36m706[0m
```

```python
model
```

```
```

**```
TFIDF
```**

`TFIDF (   normalize=True   on="body"   strip_accents=True   lowercase=True   preprocessor=None   stop_words=None   tokenizer_pattern="(?u)\b\w[\w\-]+\b"   tokenizer=None   ngram_range=(1, 1) ) `

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

```
```

```text
ROCAUC: [1;36m93.80[0m%
```

The confusion matrix:

```python
cm
```

```
```

```text
[3;91mFalse[0m   [3;92mTrue[0m  
[3;91mFalse[0m   [1;36m4[0m,[1;36m584[0m    [1;36m243[0m  
 [3;92mTrue[0m      [1;36m55[0m    [1;36m692[0m
```

```python
model
```

```
```

**```
TFIDF
```**

`TFIDF (   normalize=True   on="body"   strip_accents=True   lowercase=True   preprocessor=None   stop_words=None   tokenizer_pattern="(?u)\b\w[\w\-]+\b"   tokenizer=None   ngram_range=(1, 1) ) `

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
