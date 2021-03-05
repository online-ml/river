# Sentence classification

In this tutorial we will try to predict whether a sms is spam or not. To train our model, we will use the `SMSSpam` dataset. This dataset is unbalanced, there is only 13.4% spam. Let's look at the data:


```python
from river import datasets

datasets.SMSSpam()
```




    SMS Spam Collection dataset.
    
    The data contains 5,574 items and 1 feature (i.e. SMS body). Spam messages represent
    13.4% of the dataset. The goal is to predict whether an SMS is a spam or not.
    
          Name  SMSSpam                                                                              
          Task  Binary classification                                                                
       Samples  5,574                                                                                
      Features  1                                                                                    
        Sparse  False                                                                                
          Path  /Users/home/river_data/SMSSpam/SMSSpamCollection                                     
           URL  https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
          Size  466.71 KB                                                                            
    Downloaded  True                                                                                 




```python
from pprint import pprint

X_y = datasets.SMSSpam()

for x, y in X_y:
    
    pprint(x)
    print(f'Spam: {y}')
    break
```

    {'body': 'Go until jurong point, crazy.. Available only in bugis n great world '
             'la e buffet... Cine there got amore wat...\n'}
    Spam: False


Let's start by building a simple model like a naive bayes classifier. We will first transform the sentences with a tf-idf to feed our model. Then, we will measure the accuracy of our model with the AUC metric. This metric is adapted when the classes of our dataset are not uniformly distributed. In addition, the Naive Bayes models can perform very well on unbalanced datasets and supports multiple classes for multiple classification problems.


```python
from river import feature_extraction
from river import naive_bayes
from river import metrics

def extract_body(x):
    """Extract the body of the sms."""
    return x['body']

X_y = datasets.SMSSpam()

model = extract_body | feature_extraction.TFIDF() | naive_bayes.BernoulliNB(alpha = 0)

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




    ROCAUC: 0.929296



The confusion matrix:


```python
cm
```




               False    True
       False    4809      17
        True     103     644



The results are quite good with this first model.

Since we are working with an unbalanced dataset, we can use the `imblearn` module to artificially increase the number of spam in our dataset. For more information about the `imblearn` module, you can find a dedicated tutorial [here](https://riverml.xyz/latest/examples/imbalanced-learning/).


```python
from river import imblearn

X_y = datasets.SMSSpam()

model = extract_body | feature_extraction.TFIDF() | imblearn.RandomUnderSampler(
    classifier=naive_bayes.BernoulliNB(alpha = 0),
    desired_dist={0: 0.5, 1: 0.5},
    seed=42
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




    ROCAUC: 0.951728



The `imblearn` module improved our results. Not bad! We can visualize the pipeline to understand how the data is processed.

The confusion matrix:


```python
cm
```




               False    True
       False    4624     201
        True      41     706




```python
model.draw()
```




    
![svg](sentence_classification_files/sentence_classification_14_0.svg)
    



Now let's try to use logistic regression to classify messages. I will use different tips to make my model perform better. As in the previous example, I artificially increase the number of spam. The logistics regression will be fed from a tf-idf.


```python
from river import linear_model
from river import optim
from river import preprocessing

X_y = datasets.SMSSpam()

model = extract_body | feature_extraction.TFIDF() | preprocessing.StandardScaler() | imblearn.RandomUnderSampler(
    classifier=linear_model.LogisticRegression(optimizer=optim.SGD(0.05), loss=optim.losses.Log()),
    desired_dist={0: 0.5, 1: 0.5},
    seed=42
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




    ROCAUC: 0.834134



The confusion matrix:


```python
cm
```




               False    True
       False    4027     800
        True     124     623




```python
model.draw()
```




    
![svg](sentence_classification_files/sentence_classification_19_0.svg)
    



The results of the logistic regression are less good. It is more difficult for the model to determine if a text message is spam.

Let's try to use word embeddings to improve our logistic regression. Word embeddings allow you to represent a word as a vector. Embeddings are developed to build semantic rich vectors, i.e. the vector which represents the word **python** is close to the vector which represents the word **programming**. We will use Scipy to convert our sentence to vectors. Scipy convert a sentence to a vector by calculating the average of the embeddings of the words of the sentence.

You can download pre-trained embeddings in many languages. We will use English pre-trained embeddings as our sms are in English.

The command below allows you to download the embeddings of spacy. More informations about spacy and it's installation [here](https://spacy.io/usage).
python -m spacy download en_core_web_sm
Here, I create a custom transformer to convert an input sentence to a dict of float. I will integrate this transformer into my pipeline.


```python
import spacy

from river.base import Transformer

class Embeddings(Transformer):
    """My custom transformer, words embeddings using spacy."""
    
    def __init__(self):
        self.embeddings = spacy.load('en_core_web_sm')
        
    def transform_one(self, x, y=None):
        return {i: xi for i, xi in enumerate(self.embeddings(x).vector)}
```

Let's train our logistic regression:


```python
X_y = datasets.SMSSpam()

model = extract_body | Embeddings() | preprocessing.StandardScaler() | imblearn.RandomOverSampler(
    classifier=linear_model.LogisticRegression(loss=optim.losses.Log()),
    desired_dist={0: 0.5, 1: 0.5},
    seed=42
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




    ROCAUC: 0.926478



The confusion matrix:


```python
cm
```




               False    True
       False    4408     419
        True      45     702




```python
model.draw()
```




    
![svg](sentence_classification_files/sentence_classification_31_0.svg)
    



The results of the logistic regression with the spacy embeddings are much better than the tf-idf. We could surely improve the results by cleaning up the text. However, on this problem, the logistic regression is not better than the naive bayes model. No free lunch today.
