---
hide:
- navigation
---


# Benchmark

## Binary classification

```vegalite

{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {
    "url": "benchmarks/binary_classification.csv"
  },
  "params": [
    {
      "name": "models",
      "select": {
        "type": "point",
        "fields": [
          "model"
        ]
      },
      "bind": "legend"
    },
    {
      "name": "Dataset",
      "value": "Phishing",
      "bind": {
        "input": "select",
        "options": [
          "Phishing",
          "Bananas"
        ]
      }
    },
    {
      "name": "grid",
      "select": "interval",
      "bind": "scales"
    }
  ],
  "transform": [
    {
      "filter": {
        "field": "dataset",
        "equal": {
          "expr": "Dataset"
        }
      }
    }
  ],
  "repeat": {
    "row": [
      "Accuracy",
      "F1",
      "Memory in Mb",
      "Time in s"
    ]
  },
  "spec": {
    "width": "container",
    "mark": "line",
    "encoding": {
      "x": {
        "field": "step",
        "type": "quantitative",
        "axis": {
          "titleFontSize": 18,
          "labelFontSize": 18,
          "title": "Instance"
        }
      },
      "y": {
        "field": {
          "repeat": "row"
        },
        "type": "quantitative",
        "axis": {
          "titleFontSize": 18,
          "labelFontSize": 18
        }
      },
      "color": {
        "field": "model",
        "type": "ordinal",
        "scale": {
          "scheme": "category20b"
        },
        "title": "Models",
        "legend": {
          "titleFontSize": 18,
          "labelFontSize": 18,
          "labelLimit": 500
        }
      },
      "opacity": {
        "condition": {
          "param": "models",
          "value": 1
        },
        "value": 0.2
      }
    }
  }
}

```

### Datasets

<details>

<summary>Phishing</summary>

<pre>Phishing websites.

This dataset contains features from web pages that are classified as phishing or not.

    Name  Phishing                                                                                                     
    Task  Binary classification                                                                                        
 Samples  1,250                                                                                                        
Features  9                                                                                                            
  Sparse  False                                                                                                        
    Path  /Users/kulbach/Documents/environments/deep-river39/lib/python3.9/site-packages/river/datasets/phishing.csv.gz</pre>

</details>

<details>

<summary>Bananas</summary>

<pre>Bananas dataset.

An artificial dataset where instances belongs to several clusters with a banana shape.
There are two attributes that correspond to the x and y axis, respectively.

    Name  Bananas                                                                                                 
    Task  Binary classification                                                                                   
 Samples  5,300                                                                                                   
Features  2                                                                                                       
  Sparse  False                                                                                                   
    Path  /Users/kulbach/Documents/environments/deep-river39/lib/python3.9/site-packages/river/datasets/banana.zip</pre>

</details>

### Models

<details>

<summary>Logistic regression</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  LogisticRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.005
      )
    )
    loss=Log (
      weight_pos=1.
      weight_neg=1.
    )
    l2=0.
    l1=0.
    intercept_init=0.
    intercept_lr=Constant (
      learning_rate=0.01
    )
    clip_gradient=1e+12
    initializer=Zeros ()
  )
)</pre>

</details>

<details>

<summary>Torch MLP</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  Classifier (
    module=None
    loss_fn=&quot;binary_cross_entropy&quot;
    optimizer_fn=&lt;class 'torch.optim.adam.Adam'&gt;
    lr=0.005
    output_is_logit=True
    is_class_incremental=True
    device=&quot;cpu&quot;
    seed=42
  )
)</pre>

</details>

<details>

<summary>[baseline] Last Class</summary>

<pre>NoChangeClassifier ()</pre>

</details>

## Multiclass classification

```vegalite

{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {
    "url": "benchmarks/multiclass_classification.csv"
  },
  "params": [
    {
      "name": "models",
      "select": {
        "type": "point",
        "fields": [
          "model"
        ]
      },
      "bind": "legend"
    },
    {
      "name": "Dataset",
      "value": "ImageSegments",
      "bind": {
        "input": "select",
        "options": [
          "ImageSegments",
          "Insects",
          "Keystroke"
        ]
      }
    },
    {
      "name": "grid",
      "select": "interval",
      "bind": "scales"
    }
  ],
  "transform": [
    {
      "filter": {
        "field": "dataset",
        "equal": {
          "expr": "Dataset"
        }
      }
    }
  ],
  "repeat": {
    "row": [
      "Accuracy",
      "MicroF1",
      "MacroF1",
      "Memory in Mb",
      "Time in s"
    ]
  },
  "spec": {
    "width": "container",
    "mark": "line",
    "encoding": {
      "x": {
        "field": "step",
        "type": "quantitative",
        "axis": {
          "titleFontSize": 18,
          "labelFontSize": 18,
          "title": "Instance"
        }
      },
      "y": {
        "field": {
          "repeat": "row"
        },
        "type": "quantitative",
        "axis": {
          "titleFontSize": 18,
          "labelFontSize": 18
        }
      },
      "color": {
        "field": "model",
        "type": "ordinal",
        "scale": {
          "scheme": "category20b"
        },
        "title": "Models",
        "legend": {
          "titleFontSize": 18,
          "labelFontSize": 18,
          "labelLimit": 500
        }
      },
      "opacity": {
        "condition": {
          "param": "models",
          "value": 1
        },
        "value": 0.2
      }
    }
  }
}

```

### Datasets

<details>

<summary>ImageSegments</summary>

<pre>Image segments classification.

This dataset contains features that describe image segments into 7 classes: brickface, sky,
foliage, cement, window, path, and grass.

    Name  ImageSegments                                                                                                
    Task  Multi-class classification                                                                                   
 Samples  2,310                                                                                                        
Features  18                                                                                                           
  Sparse  False                                                                                                        
    Path  /Users/kulbach/Documents/environments/deep-river39/lib/python3.9/site-packages/river/datasets/segment.csv.zip</pre>

</details>

<details>

<summary>Insects</summary>

<pre>Insects dataset.

This dataset has different variants, which are:

- abrupt_balanced
- abrupt_imbalanced
- gradual_balanced
- gradual_imbalanced
- incremental-abrupt_balanced
- incremental-abrupt_imbalanced
- incremental-reoccurring_balanced
- incremental-reoccurring_imbalanced
- incremental_balanced
- incremental_imbalanced
- out-of-control

The number of samples and the difficulty change from one variant to another. The number of
classes is always the same (6), except for the last variant (24).

      Name  Insects                                                                                 
      Task  Multi-class classification                                                              
   Samples  52,848                                                                                  
  Features  33                                                                                      
   Classes  6                                                                                       
    Sparse  False                                                                                   
      Path  /Users/kulbach/river_data/Insects/INSECTS-abrupt_balanced_norm.arff                     
       URL  http://sites.labic.icmc.usp.br/vsouza/repository/creme/INSECTS-abrupt_balanced_norm.arff
      Size  15.66 MB                                                                                
Downloaded  True                                                                                    
   Variant  abrupt_balanced                                                                         

Parameters
----------
    variant
        Indicates which variant of the dataset to load.</pre>

</details>

<details>

<summary>Keystroke</summary>

<pre>CMU keystroke dataset.

Users are tasked to type in a password. The task is to determine which user is typing in the
password.

The only difference with the original dataset is that the &quot;sessionIndex&quot; and &quot;rep&quot; attributes
have been dropped.

      Name  Keystroke                                                     
      Task  Multi-class classification                                    
   Samples  20,400                                                        
  Features  31                                                            
    Sparse  False                                                         
      Path  /Users/kulbach/river_data/Keystroke/DSL-StrongPasswordData.csv
       URL  http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv   
      Size  4.45 MB                                                       
Downloaded  True                                                          </pre>

</details>

### Models

<details>

<summary>Torch MLP</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  Classifier (
    module=None
    loss_fn=&quot;binary_cross_entropy&quot;
    optimizer_fn=&lt;class 'torch.optim.adam.Adam'&gt;
    lr=0.005
    output_is_logit=True
    is_class_incremental=True
    device=&quot;cpu&quot;
    seed=42
  )
)</pre>

</details>

<details>

<summary>[baseline] Last Class</summary>

<pre>NoChangeClassifier ()</pre>

</details>

## Regression

```vegalite

{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {
    "url": "benchmarks/regression.csv"
  },
  "params": [
    {
      "name": "models",
      "select": {
        "type": "point",
        "fields": [
          "model"
        ]
      },
      "bind": "legend"
    },
    {
      "name": "Dataset",
      "value": "TrumpApproval",
      "bind": {
        "input": "select",
        "options": [
          "TrumpApproval",
          "Friedman7k",
          "FriedmanLEA10k",
          "FriedmanGSG10k"
        ]
      }
    },
    {
      "name": "grid",
      "select": "interval",
      "bind": "scales"
    }
  ],
  "transform": [
    {
      "filter": {
        "field": "dataset",
        "equal": {
          "expr": "Dataset"
        }
      }
    }
  ],
  "repeat": {
    "row": [
      "MAE",
      "RMSE",
      "R2",
      "Memory in Mb",
      "Time in s"
    ]
  },
  "spec": {
    "width": "container",
    "mark": "line",
    "encoding": {
      "x": {
        "field": "step",
        "type": "quantitative",
        "axis": {
          "titleFontSize": 18,
          "labelFontSize": 18,
          "title": "Instance"
        }
      },
      "y": {
        "field": {
          "repeat": "row"
        },
        "type": "quantitative",
        "axis": {
          "titleFontSize": 18,
          "labelFontSize": 18
        }
      },
      "color": {
        "field": "model",
        "type": "ordinal",
        "scale": {
          "scheme": "category20b"
        },
        "title": "Models",
        "legend": {
          "titleFontSize": 18,
          "labelFontSize": 18,
          "labelLimit": 500
        }
      },
      "opacity": {
        "condition": {
          "param": "models",
          "value": 1
        },
        "value": 0.2
      }
    }
  }
}

```

### Datasets

<details>

<summary>TrumpApproval</summary>

<pre>Donald Trump approval ratings.

This dataset was obtained by reshaping the data used by FiveThirtyEight for analyzing Donald
Trump's approval ratings. It contains 5 features, which are approval ratings collected by
5 polling agencies. The target is the approval rating from FiveThirtyEight's model. The goal of
this task is to see if we can reproduce FiveThirtyEight's model.

    Name  TrumpApproval                                                                                                      
    Task  Regression                                                                                                         
 Samples  1,001                                                                                                              
Features  6                                                                                                                  
  Sparse  False                                                                                                              
    Path  /Users/kulbach/Documents/environments/deep-river39/lib/python3.9/site-packages/river/datasets/trump_approval.csv.gz</pre>

</details>

<details>

<summary>Friedman7k</summary>

<pre>Sample from the stationary version of the Friedman dataset.

This sample contains 10k instances sampled from the Friedman generator.

    Name  Friedman7k
    Task  Regression
 Samples  7,000     
Features  10        
  Sparse  False     </pre>

</details>

<details>

<summary>FriedmanLEA10k</summary>

<pre>Sample from the FriedmanLEA generator.

This sample contains 10k instances sampled from the Friedman generator and presents
local-expanding abrupt concept drifts that locally affect the data and happen after
2k, 5k, and 8k instances.

    Name  FriedmanLEA10k
    Task  Regression    
 Samples  10,000        
Features  10            
  Sparse  False         </pre>

</details>

<details>

<summary>FriedmanGSG10k</summary>

<pre>Sample from the FriedmanGSG generator.

This sample contains 10k instances sampled from the Friedman generator and presents
global and slow gradual concept drifts that affect the data and happen after
3.5k and 7k instances. The transition window between different concepts has a length of
1k instances.

    Name  FriedmanGSG10k
    Task  Regression    
 Samples  10,000        
Features  10            
  Sparse  False         </pre>

</details>

### Models

<details>

<summary>Torch MLP</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  LinearRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.005
      )
    )
    loss=Squared ()
    l2=0.
    l1=0.
    intercept_init=0.
    intercept_lr=Constant (
      learning_rate=0.01
    )
    clip_gradient=1e+12
    initializer=Zeros ()
  )
)</pre>

</details>

# Environment

<pre>Python implementation: CPython
Python version       : 3.9.16
IPython version      : 8.7.0

river       : 0.14.0
numpy       : 1.23.5
scikit-learn: 1.0.2
pandas      : 1.3.5
scipy       : 1.9.3

Compiler    : Clang 14.0.0 (clang-1400.0.29.202)
OS          : Darwin
Release     : 22.2.0
Machine     : x86_64
Processor   : i386
CPU cores   : 16
Architecture: 64bit
</pre>

