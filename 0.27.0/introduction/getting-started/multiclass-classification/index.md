# Multi-class classification

Classification is about predicting an outcome from a fixed list of classes. The prediction is a probability distribution that assigns a probability to each possible outcome.

A labeled classification sample is made up of a bunch of features and a class. The class is usually a string or a number in the case of multiclass classification. We'll use the image segments dataset as an example.

```python
from river import datasets

dataset = datasets.ImageSegments()
dataset
```

```
```

```text
Image segments classification.

This dataset contains features that describe image segments into [1;36m7[0m classes: brickface, sky,
foliage, cement, window, path, and grass.

References
----------
[1m[[0m^[1;36m1[0m[1m][0m: [1m[[0mUCI page[1m][0m[1m([0m[4;94mhttps://archive.ics.uci.edu/ml/datasets/Statlog+[0m[4;94m([0m[4;94mImage+Segmentation[0m[4;94m)[0m[4;94m)[0m

    Name  ImageSegments                                               
    Task  Multi-class classification                                  
 Samples  [1;36m2[0m,[1;36m310[0m                                                       
Features  [1;36m18[0m                                                          
 Classes  [1;36m7[0m                                                           
  Sparse  [3;91mFalse[0m                                                       
    Path  [35m/home/runner/work/river/river/river/datasets/[0m[95msegment.csv.zip[0m
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
    [32m'region-centroid-col'[0m: [1;36m218[0m,
    [32m'region-centroid-row'[0m: [1;36m178[0m,
    [32m'short-line-density-5'[0m: [1;36m0.11111111[0m,
    [32m'short-line-density-2'[0m: [1;36m0.0[0m,
    [32m'vedge-mean'[0m: [1;36m0.8333326999999999[0m,
    [32m'vegde-sd'[0m: [1;36m0.54772234[0m,
    [32m'hedge-mean'[0m: [1;36m1.1111094[0m,
    [32m'hedge-sd'[0m: [1;36m0.5443307[0m,
    [32m'intensity-mean'[0m: [1;36m59.629630000000006[0m,
    [32m'rawred-mean'[0m: [1;36m52.44444300000001[0m,
    [32m'rawblue-mean'[0m: [1;36m75.22222[0m,
    [32m'rawgreen-mean'[0m: [1;36m51.22222[0m,
    [32m'exred-mean'[0m: [1;36m-21.555555[0m,
    [32m'exblue-mean'[0m: [1;36m46.77778[0m,
    [32m'exgreen-mean'[0m: [1;36m-25.222220999999998[0m,
    [32m'value-mean'[0m: [1;36m75.22222[0m,
    [32m'saturation-mean'[0m: [1;36m0.31899637[0m,
    [32m'hue-mean'[0m: [1;36m-2.0405545[0m
[1m}[0m
```

```python
y
```

```
```

```text
[32m'path'[0m
```

A multiclass classifier's goal is to learn how to predict a class `y` from a bunch of features `x`. We'll attempt to do this with a decision tree.

```python
from river import tree

model = tree.HoeffdingTreeClassifier()
model.predict_proba_one(x)
```

```
```

```text
[1m{[0m[1m}[0m
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

```
```

```text
[1m{[0m[32m'path'[0m: [1;36m1.0[0m[1m}[0m
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

```
```

```text
Precision   Recall   F1       Support  

brickface      [1;36m77.13[0m%   [1;36m84.85[0m%   [1;36m80.81[0m%       [1;36m330[0m  
   cement      [1;36m79.60[0m%   [1;36m83.94[0m%   [1;36m81.71[0m%       [1;36m330[0m  
  foliage      [1;36m66.36[0m%   [1;36m21.52[0m%   [1;36m32.49[0m%       [1;36m330[0m  
    grass     [1;36m100.00[0m%   [1;36m97.27[0m%   [1;36m98.62[0m%       [1;36m330[0m  
     path      [1;36m90.91[0m%   [1;36m91.19[0m%   [1;36m91.05[0m%       [1;36m329[0m  
      sky     [1;36m100.00[0m%   [1;36m98.18[0m%   [1;36m99.08[0m%       [1;36m330[0m  
   window      [1;36m43.41[0m%   [1;36m67.88[0m%   [1;36m52.96[0m%       [1;36m330[0m  

    Macro      [1;36m79.63[0m%   [1;36m77.83[0m%   [1;36m76.67[0m%            
    Micro      [1;36m77.83[0m%   [1;36m77.83[0m%   [1;36m77.83[0m%            
 Weighted      [1;36m79.62[0m%   [1;36m77.83[0m%   [1;36m76.67[0m%            

                  [1;36m77.83[0m% accuracy
```

This is a common way to evaluate an online model. In fact, there is a dedicated [`evaluate.progressive_val_score`](<https://riverml.xyz/0.27.0/api/evaluate/progressive-val-score>) function that does this for you.

```python
from river import evaluate

model = tree.HoeffdingTreeClassifier()
metric = metrics.ClassificationReport()

evaluate.progressive_val_score(dataset, model, metric)
```

```
```

```text
Precision   Recall   F1       Support  

brickface      [1;36m77.13[0m%   [1;36m84.85[0m%   [1;36m80.81[0m%       [1;36m330[0m  
   cement      [1;36m79.60[0m%   [1;36m83.94[0m%   [1;36m81.71[0m%       [1;36m330[0m  
  foliage      [1;36m66.36[0m%   [1;36m21.52[0m%   [1;36m32.49[0m%       [1;36m330[0m  
    grass     [1;36m100.00[0m%   [1;36m97.27[0m%   [1;36m98.62[0m%       [1;36m330[0m  
     path      [1;36m90.91[0m%   [1;36m91.19[0m%   [1;36m91.05[0m%       [1;36m329[0m  
      sky     [1;36m100.00[0m%   [1;36m98.18[0m%   [1;36m99.08[0m%       [1;36m330[0m  
   window      [1;36m43.41[0m%   [1;36m67.88[0m%   [1;36m52.96[0m%       [1;36m330[0m  

    Macro      [1;36m79.63[0m%   [1;36m77.83[0m%   [1;36m76.67[0m%            
    Micro      [1;36m77.83[0m%   [1;36m77.83[0m%   [1;36m77.83[0m%            
 Weighted      [1;36m79.62[0m%   [1;36m77.83[0m%   [1;36m76.67[0m%            

                  [1;36m77.83[0m% accuracy
```
