---
hide:
- navigation
---


# Benchmarks

## Environment

<pre>Python implementation: CPython
Python version       : 3.9.12
IPython version      : 7.30.1

river       : 0.10.1
numpy       : 1.22.3
scikit-learn: 1.1.0
pandas      : 1.4.1
scipy       : 1.8.0

Compiler    : Clang 12.0.1 
OS          : Darwin
Release     : 21.3.0
Machine     : x86_64
Processor   : i386
CPU cores   : 8
Architecture: 64bit
</pre>

<div>
  <link href="https://unpkg.com/tabulator-tables@5.2.6/dist/css/tabulator.min.css" rel="stylesheet">
  <script src="https://unpkg.com/tabulator-tables@5.2.6/dist/js/tabulator.min.js" type="text/javascript"></script>
</div>

<script>
        let baseColumns
        let metrics
        let columns
        </script>

## Binary classification

### Results

<div id="binary-classification-results"></div>

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
    Path  /Users/mastelini/Documents/river/river/datasets/banana.zip</pre>
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
    Path  /Users/mastelini/Documents/river/river/datasets/phishing.csv.gz</pre>
</details>

### Models

<details>
  <summary>ADWIN Bagging</summary>
  <pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>AdaBoost</summary>
  <pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Adaptive Random Forest</summary>
  <pre>[]</pre>
</details>

<details>
  <summary>Bagging</summary>
  <pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Extremely Fast Decision Tree</summary>
  <pre>ExtremelyFastDecisionTreeClassifier (
  grace_period=200
  max_depth=inf
  min_samples_reevaluate=20
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=True
  drift_window_threshold=300
  adwin_confidence=0.002
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
  <summary>Hoeffding Tree</summary>
  <pre>HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Leveraging Bagging</summary>
  <pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Logistic regression</summary>
  <pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  LogisticRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
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
  <summary>Naive Bayes</summary>
  <pre>GaussianNB ()</pre>
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
  split_confidence=1e-07
  tie_threshold=0.05
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
    distance_func=None
    softmax=False
  )
)]</pre>
</details>

<details>
  <summary>Stochastic Gradient Tree</summary>
  <pre>SGTClassifier (
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
  <summary>Streaming Random Patches</summary>
  <pre>SRPClassifier (
  model=HoeffdingTreeClassifier (
    grace_period=50
    max_depth=inf
    split_criterion=&quot;info_gain&quot;
    split_confidence=0.01
    tie_threshold=0.05
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
  )
  warning_detector=ADWIN (
    delta=0.0001
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
  split_confidence=1e-07
  tie_threshold=0.05
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
    distance_func=None
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
    distance_func=None
    softmax=False
  )
)</pre>
</details>

<script>
    var results = [{'Accuracy': 0.6257784487639177, 'Dataset': 'Bananas', 'F1': 0.4477861319966584, 'Memory': 165670, 'Model': 'ADWIN Bagging', 'Time': 2618.05}, {'Accuracy': 0.5064150943396226, 'Dataset': 'Bananas', 'F1': 0.4825949367088608, 'Memory': 3063, 'Model': 'ALMA', 'Time': 163.789}, {'Accuracy': 0.6778637478769579, 'Dataset': 'Bananas', 'F1': 0.64504054897068, 'Memory': 171980, 'Model': 'AdaBoost', 'Time': 2489.237}, {'Accuracy': 0.8856387997735422, 'Dataset': 'Bananas', 'F1': 0.8696213425129088, 'Memory': 8567570, 'Model': 'Adaptive Random Forest', 'Time': 5560.815}, {'Accuracy': 0.6335157576901302, 'Dataset': 'Bananas', 'F1': 0.45875139353400224, 'Memory': 208556, 'Model': 'Bagging', 'Time': 1883.013}, {'Accuracy': 0.6252123042083412, 'Dataset': 'Bananas', 'F1': 0.4513812154696133, 'Memory': 29602, 'Model': 'Extremely Fast Decision Tree', 'Time': 303.313}, {'Accuracy': 0.6165314210228345, 'Dataset': 'Bananas', 'F1': 0.4408365437534397, 'Memory': 43404, 'Model': 'Hoeffding Adaptive Tree', 'Time': 484.379}, {'Accuracy': 0.6421966408756369, 'Dataset': 'Bananas', 'F1': 0.5034049240440022, 'Memory': 24843, 'Model': 'Hoeffding Tree', 'Time': 251.493}, {'Accuracy': 0.8284581996603133, 'Dataset': 'Bananas', 'F1': 0.8028627195836044, 'Memory': 1244447, 'Model': 'Leveraging Bagging', 'Time': 9075.224}, {'Accuracy': 0.5373584905660377, 'Dataset': 'Bananas', 'F1': 0.22109275730622616, 'Memory': 4423, 'Model': 'Logistic regression', 'Time': 170.88}, {'Accuracy': 0.6152104170598226, 'Dataset': 'Bananas', 'F1': 0.4139120436907157, 'Memory': 3901, 'Model': 'Naive Bayes', 'Time': 240.513}, {'Accuracy': 0.8514814115870919, 'Dataset': 'Bananas', 'F1': 0.8321603753465558, 'Memory': 9988024, 'Model': 'Stacking', 'Time': 8676.398}, {'Accuracy': 0.6575471698113208, 'Dataset': 'Bananas', 'F1': 0.560639070442992, 'Memory': 2186974, 'Model': 'Stochastic Gradient Tree', 'Time': 586.001}, {'Accuracy': 0.8694093225136819, 'Dataset': 'Bananas', 'F1': 0.8508620689655172, 'Memory': 4420472, 'Model': 'Streaming Random Patches', 'Time': 11370.613}, {'Accuracy': 0.5095301000188714, 'Dataset': 'Bananas', 'F1': 0.4529572721532309, 'Memory': 535, 'Model': '[baseline] Last Class', 'Time': 75.535}, {'Accuracy': 0.8484619739573505, 'Dataset': 'Bananas', 'F1': 0.8274231678486997, 'Memory': 43826, 'Model': 'k-Nearest Neighbors', 'Time': 990.766}, {'Accuracy': 0.8935148118494796, 'Dataset': 'Phishing', 'F1': 0.8792007266121706, 'Memory': 416562, 'Model': 'ADWIN Bagging', 'Time': 1383.466}, {'Accuracy': 0.8264, 'Dataset': 'Phishing', 'F1': 0.8117953165654813, 'Memory': 4803, 'Model': 'ALMA', 'Time': 64.185}, {'Accuracy': 0.8783026421136909, 'Dataset': 'Phishing', 'F1': 0.8635547576301617, 'Memory': 293828, 'Model': 'AdaBoost', 'Time': 1524.644}, {'Accuracy': 0.9087269815852682, 'Dataset': 'Phishing', 'F1': 0.8969258589511755, 'Memory': 1460226, 'Model': 'Adaptive Random Forest', 'Time': 1276.841}, {'Accuracy': 0.8935148118494796, 'Dataset': 'Phishing', 'F1': 0.8792007266121706, 'Memory': 399544, 'Model': 'Bagging', 'Time': 1045.51}, {'Accuracy': 0.8879103282626101, 'Dataset': 'Phishing', 'F1': 0.8734177215189873, 'Memory': 132210, 'Model': 'Extremely Fast Decision Tree', 'Time': 1240.158}, {'Accuracy': 0.8670936749399519, 'Dataset': 'Phishing', 'F1': 0.8445692883895132, 'Memory': 57312, 'Model': 'Hoeffding Adaptive Tree', 'Time': 188.008}, {'Accuracy': 0.8799039231385108, 'Dataset': 'Phishing', 'F1': 0.8605947955390334, 'Memory': 43223, 'Model': 'Hoeffding Tree', 'Time': 133.039}, {'Accuracy': 0.8951160928742994, 'Dataset': 'Phishing', 'F1': 0.8783658310120707, 'Memory': 1236851, 'Model': 'Leveraging Bagging', 'Time': 4427.237}, {'Accuracy': 0.892, 'Dataset': 'Phishing', 'F1': 0.8789237668161435, 'Memory': 5811, 'Model': 'Logistic regression', 'Time': 68.847}, {'Accuracy': 0.8847077662129704, 'Dataset': 'Phishing', 'F1': 0.8714285714285714, 'Memory': 12021, 'Model': 'Naive Bayes', 'Time': 99.403}, {'Accuracy': 0.899119295436349, 'Dataset': 'Phishing', 'F1': 0.8866906474820143, 'Memory': 1692234, 'Model': 'Stacking', 'Time': 2583.277}, {'Accuracy': 0.8232, 'Dataset': 'Phishing', 'F1': 0.8141295206055509, 'Memory': 3911518, 'Model': 'Stochastic Gradient Tree', 'Time': 433.726}, {'Accuracy': 0.9095276220976781, 'Dataset': 'Phishing', 'F1': 0.8962350780532599, 'Memory': 2561969, 'Model': 'Streaming Random Patches', 'Time': 3259.081}, {'Accuracy': 0.5156124899919936, 'Dataset': 'Phishing', 'F1': 0.4474885844748858, 'Memory': 535, 'Model': '[baseline] Last Class', 'Time': 28.997}, {'Accuracy': 0.8670936749399519, 'Dataset': 'Phishing', 'F1': 0.847985347985348, 'Memory': 74814, 'Model': 'k-Nearest Neighbors', 'Time': 465.452}, {'Accuracy': 0.8903122497998399, 'Dataset': 'Phishing', 'F1': 0.8769092542677449, 'Memory': 134873, 'Model': 'Voting', 'Time': 747.333}, {'Accuracy': 0.8301566333270428, 'Dataset': 'Bananas', 'F1': 0.7949886104783599, 'Memory': 76293, 'Model': 'Voting', 'Time': 1649.353}]

    baseColumns = [
        "Dataset",
        "Model",
        "Memory",
        "Time"
    ]
    metrics = Object.keys(results[0]).filter(x => !baseColumns.includes(x)).sort();
    columns = [...baseColumns, ...metrics].map(x => ({title: x, field: x}))

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes'

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function msToTime(s) {
        function pad(n, z) {
            z = z || 2;
            return ('00' + n).slice(-z);
        }

        var ms = s % 1000;
        s = (s - ms) / 1000;
        var secs = s % 60;
        s = (s - secs) / 60;
        var mins = s % 60;
        var hrs = (s - mins) / 60;

        return pad(hrs) + ':' + pad(mins) + ':' + pad(secs) + '.' + pad(ms, 3);
    }

    columns.map((x, i) => {
        if (x.title === 'Dataset') {
            columns[i]["headerFilter"] = true
        }
        if (x.title === 'Model') {
            columns[i]["headerFilter"] = true
        }
        if (x.title === 'Memory') {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered){
                return formatBytes(cell.getValue())
            }
        }
        if (x.title === 'Time') {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return msToTime(cell.getValue())
            }
        }
        if (['Accuracy', 'F1'].includes(x.title)) {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return (100 * cell.getValue()).toFixed(2) + "%"
            }
        }
        if (['MAE', 'RMSE', 'R2'].includes(x.title)) {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return cell.getValue().toFixed(3)
            }
        }
    })

    new Tabulator('#binary-classification-results', {
        data: results,
        layout: 'fitColumns',
        columns: columns
    })
    </script>

## Multiclass classification

### Results

<div id="multiclass-classification-results"></div>

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
    Path  /Users/mastelini/Documents/river/river/datasets/segment.csv.zip</pre>
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
      Path  /Users/mastelini/river_data/Insects/INSECTS-abrupt_balanced_norm.arff                   
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
      Path  /Users/mastelini/river_data/Keystroke/DSL-StrongPasswordData.csv
       URL  http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv     
      Size  4.45 MB                                                         
Downloaded  True                                                            </pre>
</details>

### Models

<details>
  <summary>ADWIN Bagging</summary>
  <pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Adaptive Random Forest</summary>
  <pre>[]</pre>
</details>

<details>
  <summary>Bagging</summary>
  <pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Extremely Fast Decision Tree</summary>
  <pre>ExtremelyFastDecisionTreeClassifier (
  grace_period=200
  max_depth=inf
  min_samples_reevaluate=20
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
  leaf_prediction=&quot;nba&quot;
  nb_threshold=0
  nominal_attributes=None
  splitter=GaussianSplitter (
    n_splits=10
  )
  bootstrap_sampling=True
  drift_window_threshold=300
  adwin_confidence=0.002
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
  <summary>Hoeffding Tree</summary>
  <pre>HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Leveraging Bagging</summary>
  <pre>[HoeffdingTreeClassifier (
  grace_period=200
  max_depth=inf
  split_criterion=&quot;info_gain&quot;
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  split_confidence=1e-07
  tie_threshold=0.05
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
  <summary>Naive Bayes</summary>
  <pre>GaussianNB ()</pre>
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
  split_confidence=1e-07
  tie_threshold=0.05
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
    distance_func=None
    softmax=False
  )
)]</pre>
</details>

<details>
  <summary>Streaming Random Patches</summary>
  <pre>SRPClassifier (
  model=HoeffdingTreeClassifier (
    grace_period=50
    max_depth=inf
    split_criterion=&quot;info_gain&quot;
    split_confidence=0.01
    tie_threshold=0.05
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
  )
  warning_detector=ADWIN (
    delta=0.0001
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
  split_confidence=1e-07
  tie_threshold=0.05
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
    distance_func=None
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
    distance_func=None
    softmax=False
  )
)</pre>
</details>

<script>
    var results = [{'Accuracy': 0.7721957557384149, 'Dataset': 'ImageSegments', 'MacroF1': 0.7587729537473662, 'Memory': 968200, 'MicroF1': 0.772195755738415, 'Model': 'ADWIN Bagging', 'Time': 14317.314}, {'Accuracy': 0.8046773495019489, 'Dataset': 'ImageSegments', 'MacroF1': 0.7977695866822913, 'Memory': 950510, 'MicroF1': 0.8046773495019489, 'Model': 'AdaBoost', 'Time': 14164.505}, {'Accuracy': 0.8185361628410567, 'Dataset': 'ImageSegments', 'MacroF1': 0.8141343880882678, 'Memory': 1477809, 'MicroF1': 0.8185361628410566, 'Model': 'Adaptive Random Forest', 'Time': 5368.26}, {'Accuracy': 0.7769597228237333, 'Dataset': 'ImageSegments', 'MacroF1': 0.7645642360301897, 'Memory': 945619, 'MicroF1': 0.7769597228237333, 'Model': 'Bagging', 'Time': 9962.266}, {'Accuracy': 0.6253789519272412, 'Dataset': 'ImageSegments', 'MacroF1': 0.6326079461514587, 'Memory': 887128, 'MicroF1': 0.6253789519272412, 'Model': 'Extremely Fast Decision Tree', 'Time': 10523.965}, {'Accuracy': 0.7743611953226505, 'Dataset': 'ImageSegments', 'MacroF1': 0.7631658299307776, 'Memory': 107980, 'MicroF1': 0.7743611953226506, 'Model': 'Hoeffding Adaptive Tree', 'Time': 1591.331}, {'Accuracy': 0.776093546990039, 'Dataset': 'ImageSegments', 'MacroF1': 0.7631372452021825, 'Memory': 102435, 'MicroF1': 0.776093546990039, 'Model': 'Hoeffding Tree', 'Time': 1079.278}, {'Accuracy': 0.7782589865742746, 'Dataset': 'ImageSegments', 'MacroF1': 0.7660163657276378, 'Memory': 952155, 'MicroF1': 0.7782589865742745, 'Model': 'Leveraging Bagging', 'Time': 42925.328}, {'Accuracy': 0.7319185794716327, 'Dataset': 'ImageSegments', 'MacroF1': 0.7304188192194185, 'Memory': 74112, 'MicroF1': 0.7319185794716329, 'Model': 'Naive Bayes', 'Time': 564.832}, {'Accuracy': 0.8527501082719792, 'Dataset': 'ImageSegments', 'MacroF1': 0.8518698684396576, 'Memory': 1399545, 'MicroF1': 0.8527501082719792, 'Model': 'Stacking', 'Time': 7025.083}, {'Accuracy': 0.7522737115634474, 'Dataset': 'ImageSegments', 'MacroF1': 0.7487742352030357, 'Memory': 2733193, 'MicroF1': 0.7522737115634474, 'Model': 'Streaming Random Patches', 'Time': 26468.948}, {'Accuracy': 0.14811606756171503, 'Dataset': 'ImageSegments', 'MacroF1': 0.1481156678425267, 'Memory': 1436, 'MicroF1': 0.14811606756171503, 'Model': '[baseline] Last Class', 'Time': 56.817}, {'Accuracy': 0.8198354265915981, 'Dataset': 'ImageSegments', 'MacroF1': 0.8160519969700987, 'Memory': 132305, 'MicroF1': 0.8198354265915981, 'Model': 'k-Nearest Neighbors', 'Time': 1458.77}, {'Accuracy': 0.5756239710863436, 'Dataset': 'Insects', 'MacroF1': 0.5660846204171648, 'Memory': 3500308, 'MicroF1': 0.5756239710863436, 'Model': 'ADWIN Bagging', 'Time': 491606.18}, {'Accuracy': 0.5635324616345299, 'Dataset': 'Insects', 'MacroF1': 0.5546220283668154, 'Memory': 6686768, 'MicroF1': 0.5635324616345299, 'Model': 'AdaBoost', 'Time': 482156.812}, {'Accuracy': 0.7466081329119912, 'Dataset': 'Insects', 'MacroF1': 0.7443289389681618, 'Memory': 217444, 'MicroF1': 0.7466081329119912, 'Model': 'Adaptive Random Forest', 'Time': 151447.505}, {'Accuracy': 0.5730694268359604, 'Dataset': 'Insects', 'MacroF1': 0.5637706604925724, 'Memory': 5952191, 'MicroF1': 0.5730694268359604, 'Model': 'Bagging', 'Time': 338569.006}, {'Accuracy': 0.6525819819478873, 'Dataset': 'Insects', 'MacroF1': 0.6508885643449825, 'Memory': 4000179, 'MicroF1': 0.6525819819478873, 'Model': 'Extremely Fast Decision Tree', 'Time': 550728.594}, {'Accuracy': 0.6133176906919977, 'Dataset': 'Insects', 'MacroF1': 0.6061092531305883, 'Memory': 61353, 'MicroF1': 0.6133176906919977, 'Model': 'Hoeffding Adaptive Tree', 'Time': 61958.852}, {'Accuracy': 0.5373058073305959, 'Dataset': 'Insects', 'MacroF1': 0.5273644947479657, 'Memory': 625371, 'MicroF1': 0.5373058073305959, 'Model': 'Hoeffding Tree', 'Time': 35141.94}, {'Accuracy': 0.6850341552027551, 'Dataset': 'Insects', 'MacroF1': 0.6793184268681459, 'Memory': 1775086, 'MicroF1': 0.6850341552027551, 'Model': 'Leveraging Bagging', 'Time': 1219782.581}, {'Accuracy': 0.5068972694760346, 'Dataset': 'Insects', 'MacroF1': 0.4930190627831494, 'Memory': 115338, 'MicroF1': 0.5068972694760346, 'Model': 'Naive Bayes', 'Time': 18087.676}, {'Accuracy': 0.751792911612769, 'Dataset': 'Insects', 'MacroF1': 0.7498238877852431, 'Memory': 2983236, 'MicroF1': 0.7517929116127688, 'Model': 'Stacking', 'Time': 205589.696}, {'Accuracy': 0.7378091471606714, 'Dataset': 'Insects', 'MacroF1': 0.7359988196057962, 'Memory': 2531311, 'MicroF1': 0.7378091471606714, 'Model': 'Streaming Random Patches', 'Time': 1067820.552}, {'Accuracy': 0.2897610081934642, 'Dataset': 'Insects', 'MacroF1': 0.2897627257031321, 'Memory': 1454, 'MicroF1': 0.2897610081934642, 'Model': '[baseline] Last Class', 'Time': 1598.511}, {'Accuracy': 0.6868317974530248, 'Dataset': 'Insects', 'MacroF1': 0.6839236226719291, 'Memory': 227091, 'MicroF1': 0.6868317974530248, 'Model': 'k-Nearest Neighbors', 'Time': 56699.578}, {'Accuracy': 0.7196921417716555, 'Dataset': 'Keystroke', 'MacroF1': 0.721416487495366, 'Memory': 9083268, 'MicroF1': 0.7196921417716555, 'Model': 'ADWIN Bagging', 'Time': 595706.964}, {'Accuracy': 0.8415608608265112, 'Dataset': 'Keystroke', 'MacroF1': 0.8430678719218747, 'Memory': 38613262, 'MicroF1': 0.841560860826511, 'Model': 'AdaBoost', 'Time': 672862.45}, {'Accuracy': 0.9691651551546644, 'Dataset': 'Keystroke', 'MacroF1': 0.9691813964225685, 'Memory': 976550, 'MicroF1': 0.9691651551546644, 'Model': 'Adaptive Random Forest', 'Time': 33811.16}, {'Accuracy': 0.6679739202902103, 'Dataset': 'Keystroke', 'MacroF1': 0.6688529665037398, 'Memory': 10833907, 'MicroF1': 0.6679739202902103, 'Model': 'Bagging', 'Time': 514038.001}, {'Accuracy': 0.856267464091377, 'Dataset': 'Keystroke', 'MacroF1': 0.8560901018523239, 'Memory': 10480902, 'MicroF1': 0.856267464091377, 'Model': 'Extremely Fast Decision Tree', 'Time': 537981.641}, {'Accuracy': 0.729398499926467, 'Dataset': 'Keystroke', 'MacroF1': 0.7281138823431088, 'Memory': 163688, 'MicroF1': 0.7293984999264669, 'Model': 'Hoeffding Adaptive Tree', 'Time': 53354.762}, {'Accuracy': 0.6482180499044071, 'Dataset': 'Keystroke', 'MacroF1': 0.6472493759146579, 'Memory': 1142454, 'MicroF1': 0.6482180499044071, 'Model': 'Hoeffding Tree', 'Time': 51814.5}, {'Accuracy': 0.9525957154762489, 'Dataset': 'Keystroke', 'MacroF1': 0.9526888505135084, 'Memory': 1136395, 'MicroF1': 0.9525957154762489, 'Model': 'Leveraging Bagging', 'Time': 163893.914}, {'Accuracy': 0.6525319868621011, 'Dataset': 'Keystroke', 'MacroF1': 0.6515767870317882, 'Memory': 906211, 'MicroF1': 0.6525319868621011, 'Model': 'Naive Bayes', 'Time': 26760.67}, {'Accuracy': 0.9763713907544488, 'Dataset': 'Keystroke', 'MacroF1': 0.976366524785322, 'Memory': 3653482, 'MicroF1': 0.9763713907544488, 'Model': 'Stacking', 'Time': 128219.644}, {'Accuracy': 0.9494092847688612, 'Dataset': 'Keystroke', 'MacroF1': 0.9494668179502542, 'Memory': 13490486, 'MicroF1': 0.9494092847688612, 'Model': 'Streaming Random Patches', 'Time': 228579.135}, {'Accuracy': 0.9975488994558557, 'Dataset': 'Keystroke', 'MacroF1': 0.9975489582566449, 'Memory': 5287, 'MicroF1': 0.9975488994558557, 'Model': '[baseline] Last Class', 'Time': 652.881}, {'Accuracy': 0.9845090445610079, 'Dataset': 'Keystroke', 'MacroF1': 0.984507607652182, 'Memory': 224560, 'MicroF1': 0.9845090445610079, 'Model': 'k-Nearest Neighbors', 'Time': 20648.639}, {'Accuracy': 0.8033780857514076, 'Dataset': 'ImageSegments', 'MacroF1': 0.7949621132813502, 'Memory': 322233, 'MicroF1': 0.8033780857514076, 'Model': 'Voting', 'Time': 3296.878}, {'Accuracy': 0.6482297954472345, 'Dataset': 'Insects', 'MacroF1': 0.6362223941753196, 'Memory': 985469, 'MicroF1': 0.6482297954472345, 'Model': 'Voting', 'Time': 118584.905}, {'Accuracy': 0.793274180106868, 'Dataset': 'Keystroke', 'MacroF1': 0.7984237858213096, 'Memory': 2391436, 'MicroF1': 0.793274180106868, 'Model': 'Voting', 'Time': 106985.492}]

    baseColumns = [
        "Dataset",
        "Model",
        "Memory",
        "Time"
    ]
    metrics = Object.keys(results[0]).filter(x => !baseColumns.includes(x)).sort();
    columns = [...baseColumns, ...metrics].map(x => ({title: x, field: x}))

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes'

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function msToTime(s) {
        function pad(n, z) {
            z = z || 2;
            return ('00' + n).slice(-z);
        }

        var ms = s % 1000;
        s = (s - ms) / 1000;
        var secs = s % 60;
        s = (s - secs) / 60;
        var mins = s % 60;
        var hrs = (s - mins) / 60;

        return pad(hrs) + ':' + pad(mins) + ':' + pad(secs) + '.' + pad(ms, 3);
    }

    columns.map((x, i) => {
        if (x.title === 'Dataset') {
            columns[i]["headerFilter"] = true
        }
        if (x.title === 'Model') {
            columns[i]["headerFilter"] = true
        }
        if (x.title === 'Memory') {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered){
                return formatBytes(cell.getValue())
            }
        }
        if (x.title === 'Time') {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return msToTime(cell.getValue())
            }
        }
        if (['Accuracy', 'F1'].includes(x.title)) {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return (100 * cell.getValue()).toFixed(2) + "%"
            }
        }
        if (['MAE', 'RMSE', 'R2'].includes(x.title)) {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return cell.getValue().toFixed(3)
            }
        }
    })

    new Tabulator('#multiclass-classification-results', {
        data: results,
        layout: 'fitColumns',
        columns: columns
    })
    </script>

## Regression

### Results

<div id="regression-results"></div>

### Datasets

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
    Path  /Users/mastelini/Documents/river/river/datasets/trump_approval.csv.gz</pre>
</details>

### Models

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
    splitter=EBSTSplitter ()
    drift_detector=ADWIN (
      delta=0.002
    )
    alpha=0.99
    anomaly_threshold=-0.75
    m_min=30
    ordered_rule_set=True
    min_samples_split=5
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
    split_confidence=1e-07
    tie_threshold=0.05
    leaf_prediction=&quot;model&quot;
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
    splitter=EBSTSplitter ()
    min_samples_split=5
    bootstrap_sampling=True
    drift_window_threshold=300
    adwin_confidence=0.002
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
    distance_func=None
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
    splitter=EBSTSplitter ()
    drift_detector=PageHinkley (
      min_instances=30
      delta=0.005
      threshold=50
      alpha=0.9999
    )
    alpha=0.99
    anomaly_threshold=-0.75
    m_min=30
    ordered_rule_set=True
    min_samples_split=5
  )]
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
    split_confidence=1e-07
    tie_threshold=0.05
    leaf_prediction=&quot;model&quot;
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
    splitter=EBSTSplitter ()
    min_samples_split=5
    bootstrap_sampling=True
    drift_window_threshold=300
    adwin_confidence=0.002
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
  <summary>Hoeffding Tree</summary>
  <pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  HoeffdingAdaptiveTreeRegressor (
    grace_period=200
    max_depth=inf
    split_confidence=1e-07
    tie_threshold=0.05
    leaf_prediction=&quot;model&quot;
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
    splitter=EBSTSplitter ()
    min_samples_split=5
    bootstrap_sampling=True
    drift_window_threshold=300
    adwin_confidence=0.002
    binary_split=False
    max_size=500.
    memory_estimate_period=1000000
    stop_mem_management=False
    remove_poor_attrs=False
    merit_preprune=True
    seed=None
  )
)</pre>
</details>

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
  <summary>Multi-layer Perceptron</summary>
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
  <summary>Streaming Random Patches</summary>
  <pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  SRPRegressor (
    model=HoeffdingTreeRegressor (
      grace_period=50
      max_depth=inf
      split_confidence=0.01
      tie_threshold=0.05
      leaf_prediction=&quot;model&quot;
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
      splitter=EBSTSplitter ()
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
    )
    warning_detector=ADWIN (
      delta=0.0001
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
  <summary>[baseline] Mean predictor</summary>
  <pre>StatisticRegressor (
  statistic=Mean ()
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
    distance_func=None
  )
)</pre>
</details>

<script>
    var results = [{'Dataset': 'Friedman7k', 'MAE': 2.266797239461916, 'Memory': 29614147, 'Model': 'Adaptive Model Rules', 'R2': 0.6512347492119022, 'RMSE': 2.925569796880849, 'Time': 9688.85}, {'Dataset': 'Friedman7k', 'MAE': 2.0715443914050855, 'Memory': 39110662, 'Model': 'Adaptive Random Forest', 'R2': 0.7027053826578435, 'RMSE': 2.701079601225446, 'Time': 10662.744}, {'Dataset': 'Friedman7k', 'MAE': 2.322574690437379, 'Memory': 13987893, 'Model': 'Exponentially Weighted Average', 'R2': 0.6347937556123011, 'RMSE': 2.9937322603692325, 'Time': 15044.119}, {'Dataset': 'Friedman7k', 'MAE': 1.9327832857630483, 'Memory': 8273250, 'Model': 'Hoeffding Adaptive Tree', 'R2': 0.7386068659475168, 'RMSE': 2.5327419749635567, 'Time': 2502.669}, {'Dataset': 'Friedman7k', 'MAE': 2.020332317083193, 'Memory': 10966826, 'Model': 'Hoeffding Tree', 'R2': 0.7192800400505797, 'RMSE': 2.6247050487456343, 'Time': 2345.215}, {'Dataset': 'Friedman7k', 'MAE': 2.23757758349508, 'Memory': 5447, 'Model': 'Linear Regression', 'R2': 0.6549674406764808, 'RMSE': 2.9098720954400448, 'Time': 223.047}, {'Dataset': 'Friedman7k', 'MAE': 2.3552263510654856, 'Memory': 5689, 'Model': 'Linear Regression with l1 regularization', 'R2': 0.6357810009521413, 'RMSE': 2.989683112289033, 'Time': 254.21}, {'Dataset': 'Friedman7k', 'MAE': 2.5425853022401457, 'Memory': 5471, 'Model': 'Linear Regression with l2 regularization', 'R2': 0.5818589220275194, 'RMSE': 3.203356525801635, 'Time': 232.907}, {'Dataset': 'Friedman7k', 'MAE': 2.146869030920678, 'Memory': 12091, 'Model': 'Multi-layer Perceptron', 'R2': 0.6318109228679125, 'RMSE': 3.0059330965111926, 'Time': 3184.292}, {'Dataset': 'Friedman7k', 'MAE': 6.016255319658246, 'Memory': 4983, 'Model': 'Passive-Aggressive Regressor, mode 1', 'R2': -1.347670503657537, 'RMSE': 7.590361003168411, 'Time': 255.695}, {'Dataset': 'Friedman7k', 'MAE': 10.12033002328945, 'Memory': 4983, 'Model': 'Passive-Aggressive Regressor, mode 2', 'R2': -5.562891455099383, 'RMSE': 12.690872201368217, 'Time': 255.824}, {'Dataset': 'Friedman7k', 'MAE': 3.209324662171133, 'Memory': 20969242, 'Model': 'Stochastic Gradient Tree', 'R2': 0.21521551831389918, 'RMSE': 4.388529917508489, 'Time': 2329.924}, {'Dataset': 'Friedman7k', 'MAE': 1.5644434179256224, 'Memory': 68660013, 'Model': 'Streaming Random Patches', 'R2': 0.829542788638389, 'RMSE': 2.0452742452942285, 'Time': 27158.253}, {'Dataset': 'Friedman7k', 'MAE': 4.02148215121205, 'Memory': 514, 'Model': '[baseline] Mean predictor', 'R2': -0.0014847455535940135, 'RMSE': 4.957537743224416, 'Time': 98.273}, {'Dataset': 'Friedman7k', 'MAE': 2.8164610154113827, 'Memory': 78397, 'Model': 'k-Nearest Neighbors', 'R2': 0.4947712815895522, 'RMSE': 3.5211771463760955, 'Time': 1810.138}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.3742139385673555, 'Memory': 5969323, 'Model': 'Adaptive Model Rules', 'R2': 0.6239955593702382, 'RMSE': 3.057752290417493, 'Time': 12008.389}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.4022303284252065, 'Memory': 56428418, 'Model': 'Adaptive Random Forest', 'R2': 0.604885075330712, 'RMSE': 3.1344946322058447, 'Time': 16220.623}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.3899049048143155, 'Memory': 14295945, 'Model': 'Exponentially Weighted Average', 'R2': 0.6208260554075504, 'RMSE': 3.070612803488889, 'Time': 22886.661}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.10760176283711, 'Memory': 13810638, 'Model': 'Hoeffding Adaptive Tree', 'R2': 0.6874032423195169, 'RMSE': 2.7880338847278057, 'Time': 3703.04}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.1330045182145754, 'Memory': 16366574, 'Model': 'Hoeffding Tree', 'R2': 0.6831420762147579, 'RMSE': 2.8069721211121186, 'Time': 3397.315}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.2897103473066593, 'Memory': 5447, 'Model': 'Linear Regression', 'R2': 0.6487787967935958, 'RMSE': 2.9552632714772367, 'Time': 322.822}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.5037061545634054, 'Memory': 5689, 'Model': 'Linear Regression with l1 regularization', 'R2': 0.6003176271746284, 'RMSE': 3.1525596354441774, 'Time': 367.055}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.5905961448519443, 'Memory': 5471, 'Model': 'Linear Regression with l2 regularization', 'R2': 0.5766652320365953, 'RMSE': 3.244500027385861, 'Time': 347.911}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.337595204250624, 'Memory': 12091, 'Model': 'Multi-layer Perceptron', 'R2': 0.6050096867242896, 'RMSE': 3.134000314576819, 'Time': 4545.582}, {'Dataset': 'FriedmanGSG10k', 'MAE': 6.044287566711288, 'Memory': 4983, 'Model': 'Passive-Aggressive Regressor, mode 1', 'R2': -1.3389219567464345, 'RMSE': 7.6262963690899035, 'Time': 369.261}, {'Dataset': 'FriedmanGSG10k', 'MAE': 10.115920115372026, 'Memory': 4983, 'Model': 'Passive-Aggressive Regressor, mode 2', 'R2': -5.511166055168164, 'RMSE': 12.724338057614846, 'Time': 366.968}, {'Dataset': 'FriedmanGSG10k', 'MAE': 3.720464233481201, 'Memory': 27632354, 'Model': 'Stochastic Gradient Tree', 'R2': 0.042238177662453746, 'RMSE': 4.880165764242603, 'Time': 3333.67}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.0720985521286743, 'Memory': 98468277, 'Model': 'Streaming Random Patches', 'R2': 0.7039148417442476, 'RMSE': 2.7134019468515786, 'Time': 40688.375}, {'Dataset': 'FriedmanGSG10k', 'MAE': 4.056565397244311, 'Memory': 514, 'Model': '[baseline] Mean predictor', 'R2': -0.001062765801945309, 'RMSE': 4.989263800502261, 'Time': 142.935}, {'Dataset': 'FriedmanGSG10k', 'MAE': 2.875975531851087, 'Memory': 78397, 'Model': 'k-Nearest Neighbors', 'R2': 0.4857931546887805, 'RMSE': 3.5758125153022364, 'Time': 2586.037}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.4616387076740853, 'Memory': 8993671, 'Model': 'Adaptive Model Rules', 'R2': 0.6220300776226204, 'RMSE': 3.2841672209717414, 'Time': 9648.793}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.2549031139652906, 'Memory': 56216102, 'Model': 'Adaptive Random Forest', 'R2': 0.6706281169960582, 'RMSE': 3.065772347776576, 'Time': 16268.761}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.5118819600036697, 'Memory': 18419141, 'Model': 'Exponentially Weighted Average', 'R2': 0.6071824293628496, 'RMSE': 3.3480512254053356, 'Time': 22065.324}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.1377401292260303, 'Memory': 12463062, 'Model': 'Hoeffding Adaptive Tree', 'R2': 0.7116121602774277, 'RMSE': 2.8686998670541857, 'Time': 3609.104}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.162783803675369, 'Memory': 12437074, 'Model': 'Hoeffding Tree', 'R2': 0.701638258823956, 'RMSE': 2.917885244710447, 'Time': 3416.623}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.521459611350154, 'Memory': 5447, 'Model': 'Linear Regression', 'R2': 0.6037965552608275, 'RMSE': 3.3624494572844275, 'Time': 324.519}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.640726178462324, 'Memory': 5689, 'Model': 'Linear Regression with l1 regularization', 'R2': 0.5768193415984122, 'RMSE': 3.4750379066060555, 'Time': 369.549}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.804691270416271, 'Memory': 5471, 'Model': 'Linear Regression with l2 regularization', 'R2': 0.539341677314148, 'RMSE': 3.6256518778679254, 'Time': 339.582}, {'Dataset': 'FriedmanLEA10k', 'MAE': 2.4022330927696007, 'Memory': 12091, 'Model': 'Multi-layer Perceptron', 'R2': 0.6097219798337612, 'RMSE': 3.3372111657174504, 'Time': 4561.495}, {'Dataset': 'FriedmanLEA10k', 'MAE': 6.210896502344477, 'Memory': 4983, 'Model': 'Passive-Aggressive Regressor, mode 1', 'R2': -1.151193121824162, 'RMSE': 7.834952027785555, 'Time': 372.566}, {'Dataset': 'FriedmanLEA10k', 'MAE': 10.42075346727717, 'Memory': 4983, 'Model': 'Passive-Aggressive Regressor, mode 2', 'R2': -5.005052424437335, 'RMSE': 13.090464069276372, 'Time': 376.903}, {'Dataset': 'FriedmanLEA10k', 'MAE': 3.246544597341542, 'Memory': 27020122, 'Model': 'Stochastic Gradient Tree', 'R2': 0.29830631914660743, 'RMSE': 4.474766974153515, 'Time': 3255.706}, {'Dataset': 'FriedmanLEA10k', 'MAE': 1.75766823002308, 'Memory': 97572397, 'Model': 'Streaming Random Patches', 'R2': 0.7904680325003055, 'RMSE': 2.4452416632013763, 'Time': 40265.714}, {'Dataset': 'FriedmanLEA10k', 'MAE': 4.2928283270418905, 'Memory': 514, 'Model': '[baseline] Mean predictor', 'R2': -0.0009442770949354973, 'RMSE': 5.344432444452069, 'Time': 146.237}, {'Dataset': 'FriedmanLEA10k', 'MAE': 3.1012647780473994, 'Memory': 78397, 'Model': 'k-Nearest Neighbors', 'R2': 0.44465419180979704, 'RMSE': 3.9808736209297644, 'Time': 2586.896}, {'Dataset': 'TrumpApproval', 'MAE': 1.0233245255093981, 'Memory': 1932923, 'Model': 'Adaptive Model Rules', 'R2': -0.7336119671283983, 'RMSE': 2.252200758055532, 'Time': 350.808}, {'Dataset': 'TrumpApproval', 'MAE': 1.1349189151719374, 'Memory': 2410422, 'Model': 'Adaptive Random Forest', 'R2': -3.132109663766241, 'RMSE': 3.4770991244853735, 'Time': 1525.138}, {'Dataset': 'TrumpApproval', 'MAE': 40.75458054545452, 'Memory': 2792633, 'Model': 'Exponentially Weighted Average', 'R2': -567.6629514867817, 'RMSE': 40.7904615623717, 'Time': 1032.473}, {'Dataset': 'TrumpApproval', 'MAE': 1.140435026253614, 'Memory': 547010, 'Model': 'Hoeffding Adaptive Tree', 'R2': -3.7114560390466327, 'RMSE': 3.712861283297428, 'Time': 390.924}, {'Dataset': 'TrumpApproval', 'MAE': 1.0688915883104473, 'Memory': 542990, 'Model': 'Hoeffding Tree', 'R2': -3.483359889458873, 'RMSE': 3.621870793254918, 'Time': 374.42}, {'Dataset': 'TrumpApproval', 'MAE': 1.3474338935927912, 'Memory': 5215, 'Model': 'Linear Regression', 'R2': -4.81891868547912, 'RMSE': 4.126219207359161, 'Time': 27.848}, {'Dataset': 'TrumpApproval', 'MAE': 1.2151577407875496, 'Memory': 5457, 'Model': 'Linear Regression with l1 regularization', 'R2': -4.650904180700232, 'RMSE': 4.0662129936129725, 'Time': 30.516}, {'Dataset': 'TrumpApproval', 'MAE': 1.9978419034436667, 'Memory': 5239, 'Model': 'Linear Regression with l2 regularization', 'R2': -5.640263007309195, 'RMSE': 4.407819407941372, 'Time': 27.93}, {'Dataset': 'TrumpApproval', 'MAE': 1.5898274221188347, 'Memory': 11583, 'Model': 'Multi-layer Perceptron', 'R2': -8.045077068340989, 'RMSE': 5.144430305753038, 'Time': 389.321}, {'Dataset': 'TrumpApproval', 'MAE': 4.903983530526025, 'Memory': 4651, 'Model': 'Passive-Aggressive Regressor, mode 1', 'R2': -14.171985226958702, 'RMSE': 6.662732200837991, 'Time': 31.427}, {'Dataset': 'TrumpApproval', 'MAE': 31.12616606921402, 'Memory': 4651, 'Model': 'Passive-Aggressive Regressor, mode 2', 'R2': -403.916378910996, 'RMSE': 34.42023446743753, 'Time': 30.932}, {'Dataset': 'TrumpApproval', 'MAE': 9.429746533156267, 'Memory': 2116974, 'Model': 'Stochastic Gradient Tree', 'R2': -108.97151968967047, 'RMSE': 17.937886241411594, 'Time': 158.125}, {'Dataset': 'TrumpApproval', 'MAE': 1.1714913650238603, 'Memory': 1800701, 'Model': 'Streaming Random Patches', 'R2': -1.6041644181648174, 'RMSE': 2.7603576294334173, 'Time': 4328.816}, {'Dataset': 'TrumpApproval', 'MAE': 1.567554989468773, 'Memory': 514, 'Model': '[baseline] Mean predictor', 'R2': -0.6584830635688459, 'RMSE': 2.202858861923226, 'Time': 12.714}, {'Dataset': 'TrumpApproval', 'MAE': 0.49369847918747883, 'Memory': 69121, 'Model': 'k-Nearest Neighbors', 'R2': 0.22347386899695654, 'RMSE': 1.5073329387274894, 'Time': 186.799}]

    baseColumns = [
        "Dataset",
        "Model",
        "Memory",
        "Time"
    ]
    metrics = Object.keys(results[0]).filter(x => !baseColumns.includes(x)).sort();
    columns = [...baseColumns, ...metrics].map(x => ({title: x, field: x}))

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes'

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function msToTime(s) {
        function pad(n, z) {
            z = z || 2;
            return ('00' + n).slice(-z);
        }

        var ms = s % 1000;
        s = (s - ms) / 1000;
        var secs = s % 60;
        s = (s - secs) / 60;
        var mins = s % 60;
        var hrs = (s - mins) / 60;

        return pad(hrs) + ':' + pad(mins) + ':' + pad(secs) + '.' + pad(ms, 3);
    }

    columns.map((x, i) => {
        if (x.title === 'Dataset') {
            columns[i]["headerFilter"] = true
        }
        if (x.title === 'Model') {
            columns[i]["headerFilter"] = true
        }
        if (x.title === 'Memory') {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered){
                return formatBytes(cell.getValue())
            }
        }
        if (x.title === 'Time') {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return msToTime(cell.getValue())
            }
        }
        if (['Accuracy', 'F1'].includes(x.title)) {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return (100 * cell.getValue()).toFixed(2) + "%"
            }
        }
        if (['MAE', 'RMSE', 'R2'].includes(x.title)) {
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {
                return cell.getValue().toFixed(3)
            }
        }
    })

    new Tabulator('#regression-results', {
        data: results,
        layout: 'fitColumns',
        columns: columns
    })
    </script>

