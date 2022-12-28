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
      "value": "Bananas",
      "bind": {
        "input": "select",
        "options": [
          "Bananas",
          "Elec2",
          "Phishing",
          "SMTP"
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

<summary>Bananas</summary>

<pre>Bananas dataset.

An artificial dataset where instances belongs to several clusters with a banana shape.
There are two attributes that correspond to the x and y axis, respectively.

    Name  Bananas                                               
    Task  Binary classification                                 
 Samples  5,300                                                 
Features  2                                                     
  Sparse  False                                                 
    Path  /home/kulbach/projects/river/river/datasets/banana.zip</pre>

</details>

<details>

<summary>Elec2</summary>

<pre>Electricity prices in New South Wales.

This is a binary classification task, where the goal is to predict if the price of electricity
will go up or down.

This data was collected from the Australian New South Wales Electricity Market. In this market,
prices are not fixed and are affected by demand and supply of the market. They are set every
five minutes. Electricity transfers to/from the neighboring state of Victoria were done to
alleviate fluctuations.

      Name  Elec2                                                      
      Task  Binary classification                                      
   Samples  45,312                                                     
  Features  8                                                          
    Sparse  False                                                      
      Path  /home/kulbach/river_data/Elec2/electricity.csv             
       URL  https://maxhalford.github.io/files/datasets/electricity.zip
      Size  2.95 MB                                                    
Downloaded  True                                                       </pre>

</details>

<details>

<summary>Phishing</summary>

<pre>Phishing websites.

This dataset contains features from web pages that are classified as phishing or not.

    Name  Phishing                                                   
    Task  Binary classification                                      
 Samples  1,250                                                      
Features  9                                                          
  Sparse  False                                                      
    Path  /home/kulbach/projects/river/river/datasets/phishing.csv.gz</pre>

</details>

<details>

<summary>SMTP</summary>

<pre>SMTP dataset from the KDD 1999 cup.

The goal is to predict whether or not an SMTP connection is anomalous or not. The dataset only
contains 2,211 (0.4%) positive labels.

      Name  SMTP                                                
      Task  Binary classification                               
   Samples  95,156                                              
  Features  3                                                   
    Sparse  False                                               
      Path  /home/kulbach/river_data/SMTP/smtp.csv              
       URL  https://maxhalford.github.io/files/datasets/smtp.zip
      Size  5.23 MB                                             
Downloaded  True                                                </pre>

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

<summary>ALMA</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  ALMAClassifier (
    p=2
    alpha=0.9
    B=1.111111
    C=1.414214
  )
)</pre>

</details>

<details>

<summary>sklearn SGDClassifier</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  SKL2RiverClassifier (
    estimator=SGDClassifier(eta0=0.005, learning_rate='constant', loss='log', penalty='none')
    classes=[False, True]
  )
)</pre>

</details>

<details>

<summary>Vowpal Wabbit logistic regression</summary>

<pre>VW2RiverClassifier ()</pre>

</details>

<details>

<summary>Naive Bayes</summary>

<pre>GaussianNB ()</pre>

</details>

<details>

<summary>Hoeffding Tree</summary>

<pre>HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)</pre>

</details>

<details>

<summary>Hoeffding Adaptive Tree</summary>

<pre>HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=True
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=42
)</pre>

</details>

<details>

<summary>Adaptive Random Forest</summary>

<pre>[]</pre>

</details>

<details>

<summary>Streaming Random Patches</summary>

<pre>SRPClassifier (
  model=HoeffdingTreeClassifier (
    grace_period=50
    max_depth=inf
    split_criterion=&quot;info_gain&quot;
    delta=0.01
    tau=0.05
    leaf_prediction=&quot;nba&quot;
    nb_threshold=0
    nominal_attributes=None
    splitter=GaussianSplitter (
      n_splits=10
    )
    binary_split=False
    max_size=100.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
  )
  n_models=10
  subspace_size=0.6
  training_method=&quot;patches&quot;
  lam=6
  drift_detector=ADWIN (
    delta=1e-05
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  warning_detector=ADWIN (
    delta=0.0001
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  disable_detector=&quot;off&quot;
  disable_weighted_vote=False
  seed=None
  metric=Accuracy (
    cm=ConfusionMatrix (
      classes=[]
    )
  )
)</pre>

</details>

<details>

<summary>k-Nearest Neighbors</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  KNNClassifier (
    n_neighbors=5
    window_size=100
    min_distance_keep=0.
    weighted=True
    cleanup_every=0
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
    softmax=False
  )
)</pre>

</details>

<details>

<summary>ADWIN Bagging</summary>

<pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)]</pre>

</details>

<details>

<summary>AdaBoost</summary>

<pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)]</pre>

</details>

<details>

<summary>Bagging</summary>

<pre>[HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
)]</pre>

</details>

<details>

<summary>Leveraging Bagging</summary>

<pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)]</pre>

</details>

<details>

<summary>Stacking</summary>

<pre>[Pipeline (
  StandardScaler (
    with_std=True
  ),
  SoftmaxRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
      )
    )
    loss=CrossEntropy (
      class_weight={}
    )
    l2=0
  )
), GaussianNB (), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), Pipeline (
  StandardScaler (
    with_std=True
  ),
  KNNClassifier (
    n_neighbors=5
    window_size=100
    min_distance_keep=0.
    weighted=True
    cleanup_every=0
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
    softmax=False
  )
)]</pre>

</details>

<details>

<summary>Voting</summary>

<pre>VotingClassifier (
  models=[Pipeline (
  StandardScaler (
    with_std=True
  ),
  SoftmaxRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
      )
    )
    loss=CrossEntropy (
      class_weight={}
    )
    l2=0
  )
), GaussianNB (), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), Pipeline (
  StandardScaler (
    with_std=True
  ),
  KNNClassifier (
    n_neighbors=5
    window_size=100
    min_distance_keep=0.
    weighted=True
    cleanup_every=0
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
    softmax=False
  )
)]
  use_probabilities=True
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
    Path  /home/kulbach/projects/river/river/datasets/segment.csv.zip</pre>

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
      Path  /home/kulbach/river_data/Insects/INSECTS-abrupt_balanced_norm.arff                      
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
      Path  /home/kulbach/river_data/Keystroke/DSL-StrongPasswordData.csv
       URL  http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv  
      Size  4.45 MB                                                      
Downloaded  True                                                         </pre>

</details>

### Models

<details>

<summary>Naive Bayes</summary>

<pre>GaussianNB ()</pre>

</details>

<details>

<summary>Hoeffding Tree</summary>

<pre>HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)</pre>

</details>

<details>

<summary>Hoeffding Adaptive Tree</summary>

<pre>HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=True
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=42
)</pre>

</details>

<details>

<summary>Adaptive Random Forest</summary>

<pre>[]</pre>

</details>

<details>

<summary>Streaming Random Patches</summary>

<pre>SRPClassifier (
  model=HoeffdingTreeClassifier (
    grace_period=50
    max_depth=inf
    split_criterion=&quot;info_gain&quot;
    delta=0.01
    tau=0.05
    leaf_prediction=&quot;nba&quot;
    nb_threshold=0
    nominal_attributes=None
    splitter=GaussianSplitter (
      n_splits=10
    )
    binary_split=False
    max_size=100.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
  )
  n_models=10
  subspace_size=0.6
  training_method=&quot;patches&quot;
  lam=6
  drift_detector=ADWIN (
    delta=1e-05
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  warning_detector=ADWIN (
    delta=0.0001
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  disable_detector=&quot;off&quot;
  disable_weighted_vote=False
  seed=None
  metric=Accuracy (
    cm=ConfusionMatrix (
      classes=[]
    )
  )
)</pre>

</details>

<details>

<summary>k-Nearest Neighbors</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  KNNClassifier (
    n_neighbors=5
    window_size=100
    min_distance_keep=0.
    weighted=True
    cleanup_every=0
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
    softmax=False
  )
)</pre>

</details>

<details>

<summary>ADWIN Bagging</summary>

<pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)]</pre>

</details>

<details>

<summary>AdaBoost</summary>

<pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)]</pre>

</details>

<details>

<summary>Bagging</summary>

<pre>[HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
), HoeffdingAdaptiveTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=False
  drift_window_threshold=300
  drift_detector=ADWIN (
    delta=0.002
    clock=32
    max_buckets=5
    min_window_length=5
    grace_period=10
  )
  switch_significance=0.05
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
  seed=None
)]</pre>

</details>

<details>

<summary>Leveraging Bagging</summary>

<pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
)]</pre>

</details>

<details>

<summary>Stacking</summary>

<pre>[Pipeline (
  StandardScaler (
    with_std=True
  ),
  SoftmaxRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
      )
    )
    loss=CrossEntropy (
      class_weight={}
    )
    l2=0
  )
), GaussianNB (), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), Pipeline (
  StandardScaler (
    with_std=True
  ),
  KNNClassifier (
    n_neighbors=5
    window_size=100
    min_distance_keep=0.
    weighted=True
    cleanup_every=0
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
    softmax=False
  )
)]</pre>

</details>

<details>

<summary>Voting</summary>

<pre>VotingClassifier (
  models=[Pipeline (
  StandardScaler (
    with_std=True
  ),
  SoftmaxRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
      )
    )
    loss=CrossEntropy (
      class_weight={}
    )
    l2=0
  )
), GaussianNB (), HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  delta=1e-07
  tau=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  binary_split=False
  max_size=100.
  memory_estimate_period=1000000
  stop_mem_management=False
  remove_poor_attrs=False
  merit_preprune=True
), Pipeline (
  StandardScaler (
    with_std=True
  ),
  KNNClassifier (
    n_neighbors=5
    window_size=100
    min_distance_keep=0.
    weighted=True
    cleanup_every=0
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
    softmax=False
  )
)]
  use_probabilities=True
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
      "value": "ChickWeights",
      "bind": {
        "input": "select",
        "options": [
          "ChickWeights",
          "TrumpApproval"
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

<summary>ChickWeights</summary>

<pre>Chick weights along time.

The stream contains 578 items and 3 features. The goal is to predict the weight of each chick
along time, according to the diet the chick is on. The data is ordered by time and then by
chick.

    Name  ChickWeights                                                 
    Task  Regression                                                   
 Samples  578                                                          
Features  3                                                            
  Sparse  False                                                        
    Path  /home/kulbach/projects/river/river/datasets/chick-weights.csv</pre>

</details>

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
    Path  /home/kulbach/projects/river/river/datasets/trump_approval.csv.gz</pre>

</details>

### Models

<details>

<summary>Linear Regression</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  LinearRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
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

<details>

<summary>Linear Regression with l1 regularization</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  LinearRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
      )
    )
    loss=Squared ()
    l2=0.
    l1=1.
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

<summary>Linear Regression with l2 regularization</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  LinearRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
      )
    )
    loss=Squared ()
    l2=1.
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

<summary>Passive-Aggressive Regressor, mode 1</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  PARegressor (
    C=1.
    mode=1
    eps=0.1
    learn_intercept=True
  )
)</pre>

</details>

<details>

<summary>Passive-Aggressive Regressor, mode 2</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  PARegressor (
    C=1.
    mode=2
    eps=0.1
    learn_intercept=True
  )
)</pre>

</details>

<details>

<summary>k-Nearest Neighbors</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  KNNRegressor (
    n_neighbors=5
    window_size=100
    aggregation_method=&quot;mean&quot;
    min_distance_keep=0.
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
  )
)</pre>

</details>

<details>

<summary>Hoeffding Tree</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  HoeffdingTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
  )
)</pre>

</details>

<details>

<summary>Hoeffding Adaptive Tree</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=True
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=42
  )
)</pre>

</details>

<details>

<summary>Stochastic Gradient Tree</summary>

<pre>SGTRegressor (
  delta=1e-07
  grace_period=200
  init_pred=0.
  max_depth=inf
  lambda_value=0.1
  gamma=1.
  nominal_attributes=[]
  feature_quantizer=StaticQuantizer (
    n_bins=64
    warm_start=100
    buckets=None
  )
)</pre>

</details>

<details>

<summary>Adaptive Random Forest</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  []
)</pre>

</details>

<details>

<summary>Adaptive Model Rules</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  AMRules (
    n_min=200
    delta=1e-07
    tau=0.05
    pred_type=&quot;adaptive&quot;
    pred_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    splitter=TEBSTSplitter (
      digits=1
    )
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    fading_factor=0.99
    anomaly_threshold=-0.75
    m_min=30
    ordered_rule_set=True
    min_samples_split=5
  )
)</pre>

</details>

<details>

<summary>Streaming Random Patches</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  SRPRegressor (
    model=HoeffdingTreeRegressor (
      grace_period=50
      max_depth=inf
      delta=0.01
      tau=0.05
      leaf_prediction=&quot;adaptive&quot;
      leaf_model=LinearRegression (
        optimizer=SGD (
          lr=Constant (
            learning_rate=0.01
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
      model_selector_decay=0.95
      nominal_attributes=None
      splitter=TEBSTSplitter (
        digits=1
      )
      min_samples_split=5
      binary_split=False
      max_size=500.
      memory_estimate_period=1000000
      stop_mem_management=False
      remove_poor_attrs=False
      merit_preprune=True
    )
    n_models=10
    subspace_size=0.6
    training_method=&quot;patches&quot;
    lam=6
    drift_detector=ADWIN (
      delta=1e-05
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    warning_detector=ADWIN (
      delta=0.0001
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    disable_detector=&quot;off&quot;
    disable_weighted_vote=True
    drift_detection_criteria=&quot;error&quot;
    aggregation_method=&quot;mean&quot;
    seed=42
    metric=MAE ()
  )
)</pre>

</details>

<details>

<summary>Bagging</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  [HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=False
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  )]
)</pre>

</details>

<details>

<summary>Exponentially Weighted Average</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  [LinearRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
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
  ), HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    delta=1e-07
    tau=0.05
    leaf_prediction=&quot;adaptive&quot;
    leaf_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    model_selector_decay=0.95
    nominal_attributes=None
    splitter=TEBSTSplitter (
      digits=1
    )
    min_samples_split=5
    bootstrap_sampling=True
    drift_window_threshold=300
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    switch_significance=0.05
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  ), KNNRegressor (
    n_neighbors=5
    window_size=100
    aggregation_method=&quot;mean&quot;
    min_distance_keep=0.
    distance_func=functools.partial(&lt;function minkowski_distance at 0x7f2d38a59ea0&gt;, p=2)
  ), AMRules (
    n_min=200
    delta=1e-07
    tau=0.05
    pred_type=&quot;adaptive&quot;
    pred_model=LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
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
    splitter=TEBSTSplitter (
      digits=1
    )
    drift_detector=ADWIN (
      delta=0.002
      clock=32
      max_buckets=5
      min_window_length=5
      grace_period=10
    )
    fading_factor=0.99
    anomaly_threshold=-0.75
    m_min=30
    ordered_rule_set=True
    min_samples_split=5
  )]
)</pre>

</details>

<details>

<summary>River MLP</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  MLPRegressor (
    hidden_dims=(5,)
    activations=(&lt;class 'river.neural_net.activations.ReLU'&gt;, &lt;class 'river.neural_net.activations.ReLU'&gt;, &lt;class 'river.neural_net.activations.Identity'&gt;)
    loss=Squared ()
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.001
      )
    )
    seed=42
  )
)</pre>

</details>

<details>

<summary>[baseline] Mean predictor</summary>

<pre>StatisticRegressor (
  statistic=Mean ()
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

