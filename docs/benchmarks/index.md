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
    "url": "benchmarks/Binary classification.csv"
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
      "Memory",
      "Time"
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
    Path  /Users/kulbach/Documents/projects/IncrementalLearning/river/river/datasets/banana.zip</pre>

</details>

<details>

<summary>CreditCard</summary>

<pre>Credit card frauds.

The datasets contains transactions made by credit cards in September 2013 by european
cardholders. This dataset presents transactions that occurred in two days, where we have 492
frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class
(frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation.
Unfortunately, due to confidentiality issues, we cannot provide the original features and more
background information about the data. Features V1, V2, ... V28 are the principal components
obtained with PCA, the only features which have not been transformed with PCA are 'Time' and
'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first
transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be
used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and
it takes value 1 in case of fraud and 0 otherwise.

      Name  CreditCard                                                     
      Task  Binary classification                                          
   Samples  284,807                                                        
  Features  30                                                             
    Sparse  False                                                          
      Path  /Users/kulbach/river_data/CreditCard/creditcard.csv            
       URL  https://maxhalford.github.io/files/datasets/creditcardfraud.zip
      Size  143.84 MB                                                      
Downloaded  True                                                           </pre>

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
      Path  /Users/kulbach/river_data/Elec2/electricity.csv            
       URL  https://maxhalford.github.io/files/datasets/electricity.zip
      Size  2.95 MB                                                    
Downloaded  True                                                       </pre>

</details>

<details>

<summary>Higgs</summary>

<pre>Higgs dataset.

The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22)
are kinematic properties measured by the particle detectors in the accelerator. The last seven
features are functions of the first 21 features; these are high-level features derived by
physicists to help discriminate between the two classes.

      Name  Higgs                                                                       
      Task  Binary classification                                                       
   Samples  11,000,000                                                                  
  Features  28                                                                          
    Sparse  False                                                                       
      Path  /Users/kulbach/river_data/Higgs/HIGGS.csv.gz                                
       URL  https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
      Size  2.62 GB                                                                     
Downloaded  False                                                                       </pre>

</details>

<details>

<summary>HTTP</summary>

<pre>HTTP dataset of the KDD 1999 cup.

The goal is to predict whether or not an HTTP connection is anomalous or not. The dataset only
contains 2,211 (0.4%) positive labels.

      Name  HTTP                                                      
      Task  Binary classification                                     
   Samples  567,498                                                   
  Features  3                                                         
    Sparse  False                                                     
      Path  /Users/kulbach/river_data/HTTP/kdd99_http.csv             
       URL  https://maxhalford.github.io/files/datasets/kdd99_http.zip
      Size  30.9 MB                                                   
Downloaded  False                                                     </pre>

</details>

<details>

<summary>MaliciousURL</summary>

<pre>Malicious URLs dataset.

This dataset contains features about URLs that are classified as malicious or not.

      Name  MaliciousURL                                               
      Task  Binary classification                                      
   Samples  2,396,130                                                  
  Features  3,231,961                                                  
    Sparse  True                                                       
      Path  /Users/kulbach/river_data/MaliciousURL/url_svmlight        
       URL  http://www.sysnet.ucsd.edu/projects/url/url_svmlight.tar.gz
      Size  2.06 GB                                                    
Downloaded  False                                                      </pre>

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
    Path  /Users/kulbach/Documents/projects/IncrementalLearning/river/river/datasets/phishing.csv.gz</pre>

</details>

<details>

<summary>SMSSpam</summary>

<pre>SMS Spam Collection dataset.

The data contains 5,574 items and 1 feature (i.e. SMS body). Spam messages represent
13.4% of the dataset. The goal is to predict whether an SMS is a spam or not.

      Name  SMSSpam                                                                              
      Task  Binary classification                                                                
   Samples  5,574                                                                                
  Features  1                                                                                    
    Sparse  False                                                                                
      Path  /Users/kulbach/river_data/SMSSpam/SMSSpamCollection                                  
       URL  https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
      Size  466.71 KB                                                                            
Downloaded  False                                                                                </pre>

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
      Path  /Users/kulbach/river_data/SMTP/smtp.csv             
       URL  https://maxhalford.github.io/files/datasets/smtp.zip
      Size  5.23 MB                                             
Downloaded  False                                               </pre>

</details>

<details>

<summary>TREC07</summary>

<pre>TREC's 2007 Spam Track dataset.

The data contains 75,419 chronologically ordered items, i.e. 3 months of emails delivered
to a particular server in 2007. Spam messages represent 66.6% of the dataset.
The goal is to predict whether an email is a spam or not.

The available raw features are: sender, recipients, date, subject, body.

      Name  TREC07                                                 
      Task  Binary classification                                  
   Samples  75,419                                                 
  Features  5                                                      
    Sparse  False                                                  
      Path  /Users/kulbach/river_data/TREC07/trec07p.csv           
       URL  https://maxhalford.github.io/files/datasets/trec07p.zip
      Size  137.81 MB                                              
Downloaded  False                                                  </pre>

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

<summary>sklearn SGDClassifier</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  SKL2RiverClassifier (
    estimator=SGDClassifier(eta0=0.005, learning_rate='constant', loss='log_loss',
                penalty='none')
    classes=[False, True]
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
    optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
    lr=0.005
    output_is_logit=True
    is_class_incremental=False
    device=&quot;cpu&quot;
    seed=42
  )
)</pre>

</details>

<details>

<summary>Torch LogReg</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  Classifier (
    module=None
    loss_fn=&quot;binary_cross_entropy&quot;
    optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
    lr=0.005
    output_is_logit=True
    is_class_incremental=False
    device=&quot;cpu&quot;
    seed=42
  )
)</pre>

</details>

<details>

<summary>Torch LSTM</summary>

<pre>Pipeline (
  StandardScaler (
    with_std=True
  ),
  RollingClassifier (
    module=None
    loss_fn=&quot;binary_cross_entropy&quot;
    optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
    lr=0.005
    output_is_logit=True
    is_class_incremental=False
    device=&quot;cpu&quot;
    seed=42
    window_size=20
    append_predict=False
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

<summary>Extremely Fast Decision Tree</summary>

<pre>ExtremelyFastDecisionTreeClassifier (
  grace_period=200
  max_depth=inf
  min_samples_reevaluate=20
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
    distance_func=functools.partial(&lt;function minkowski_distance at 0x13d12cf70&gt;, p=2)
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
    distance_func=functools.partial(&lt;function minkowski_distance at 0x13d12cf70&gt;, p=2)
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
    distance_func=functools.partial(&lt;function minkowski_distance at 0x13d12cf70&gt;, p=2)
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

