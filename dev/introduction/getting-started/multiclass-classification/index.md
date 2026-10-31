# Multi-class classification

Classification is about predicting an outcome from a fixed list of classes. The prediction is a probability distribution that assigns a probability to each possible outcome.

A labeled classification sample is made up of a bunch of features and a class. The class is usually a string or a number in the case of multiclass classification. We'll use the image segments dataset as an example.

```python
from river import datasets

dataset = datasets.ImageSegments()
dataset
```

```text
Image segments classification.

This dataset contains features that describe image segments into 7 classes: brickface, sky,
foliage, cement, window, path, and grass.

References
----------
[^1]: [UCI page](https://archive.ics.uci.edu/ml/datasets/Statlog+(Image+Segmentation))

    Name  ImageSegments                                           
    Task  Multi-class classification                              
 Samples  2,310                                                   
Features  18                                                      
 Classes  7                                                       
  Sparse  False                                                   
    Path  /Users/max/projects/river/river/datasets/segment.csv.zip
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
{'region-centroid-col': 218,
 'region-centroid-row': 178,
 'short-line-density-5': 0.11111111,
 'short-line-density-2': 0.0,
 'vedge-mean': 0.8333326999999999,
 'vegde-sd': 0.54772234,
 'hedge-mean': 1.1111094,
 'hedge-sd': 0.5443307,
 'intensity-mean': 59.629630000000006,
 'rawred-mean': 52.44444300000001,
 'rawblue-mean': 75.22222,
 'rawgreen-mean': 51.22222,
 'exred-mean': -21.555555,
 'exblue-mean': 46.77778,
 'exgreen-mean': -25.222220999999998,
 'value-mean': 75.22222,
 'saturation-mean': 0.31899637,
 'hue-mean': -2.0405545}
```

```python
y
```

```text
'path'
```

A multiclass classifier's goal is to learn how to predict a class `y` from a bunch of features `x`. We'll attempt to do this with a decision tree.

```python
from river import tree

model = tree.HoeffdingTreeClassifier()
model.predict_proba_one(x)
```

```text
{}
```

The reason why the output dictionary is empty is because the model hasn't seen any data yet. It isn't aware of the dataset whatsoever. If this were a binary classifier, then it would output a probability of 50% for `True` and `False` because the classes are implicit. But in this case we're doing multiclass classification.

Likewise, the `predict_one` method initially returns `None` because the model hasn't seen any labeled data yet.

```python
print(model.predict_one(x))
```

```text
None
```

If we update the model and try again, then we see that a probability of 100% is assigned to the `'path'` class because that's the only one the model is aware of.

```python
model.learn_one(x, y)
model.predict_proba_one(x)
```

```text
{'path': 1.0}
```

This is a strength of online classifiers: they're able to deal with new classes appearing in the data stream.

Typically, an online model makes a prediction, and then learns once the ground truth reveals itself. The prediction and the ground truth can be compared to measure the model's correctness. If you have a dataset available, you can loop over it, make a prediction, update the model, and compare the model's output with the ground truth. This is called progressive validation.

```python
from river import metrics

model = tree.HoeffdingTreeClassifier()

metric = metrics.ClassificationReport()

for x, y in dataset:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    if y_pred is not None:
        metric.update(y, y_pred)

metric
```

```text
Precision   Recall   F1       Support  

brickface      77.13%   84.85%   80.81%       330  
   cement      79.60%   83.94%   81.71%       330  
  foliage      66.36%   21.52%   32.49%       330  
    grass     100.00%   97.27%   98.62%       330  
     path      90.91%   91.19%   91.05%       329  
      sky     100.00%   98.18%   99.08%       330  
   window      43.41%   67.88%   52.96%       330  

    Macro      79.63%   77.83%   76.67%            
    Micro      77.83%   77.83%   77.83%            
 Weighted      79.62%   77.83%   76.67%            

                  77.83% accuracy
```

This is a common way to evaluate an online model. In fact, there is a dedicated [`evaluate.progressive_val_score`](<https://riverml.xyz/dev/api/evaluate/progressive-val-score>) function that does this for you.

```python
from river import evaluate

model = tree.HoeffdingTreeClassifier()
metric = metrics.ClassificationReport()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
Precision   Recall   F1       Support  

brickface      77.13%   84.85%   80.81%       330  
   cement      79.60%   83.94%   81.71%       330  
  foliage      66.36%   21.52%   32.49%       330  
    grass     100.00%   97.27%   98.62%       330  
     path      90.91%   91.19%   91.05%       329  
      sky     100.00%   98.18%   99.08%       330  
   window      43.41%   67.88%   52.96%       330  

    Macro      79.63%   77.83%   76.67%            
    Micro      77.83%   77.83%   77.83%            
 Weighted      79.62%   77.83%   76.67%            

                  77.83% accuracy
```
