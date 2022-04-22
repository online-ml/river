# Pipelines

Pipelines are an integral part of river. We encourage their usage and apply them in many of their examples.

The `compose.Pipeline` contains all the logic for building and applying pipelines. A pipeline is essentially a list of estimators that are applied in sequence. The only requirement is that the first `n - 1` steps be transformers. The last step can be a regressor, a classifier, a clusterer, a transformer, etc. Here is an example:


```python
from river import compose
from river import linear_model
from river import preprocessing
from river import feature_extraction

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    feature_extraction.PolynomialExtender(),
    linear_model.LinearRegression()
)
```

You can also use the `|` operator, as so:


```python
model = (
    preprocessing.StandardScaler() |
    feature_extraction.PolynomialExtender() |
    linear_model.LinearRegression()
)
```

Or, equally:


```python
model = preprocessing.StandardScaler() 
model |= feature_extraction.PolynomialExtender()
model |= linear_model.LinearRegression()
```

A pipeline has a `draw` method that can be used to visualize it:


```python
model
```




<div><div class="component pipeline"><details class="component estimator"><summary><pre class="estimator-name">StandardScaler</pre></summary><code class="estimator-params">
{'counts': Counter(),
 'means': defaultdict(&lt;class 'float'&gt;, {}),
 'vars': defaultdict(&lt;class 'float'&gt;, {}),
 'with_std': True}

</code></details><details class="component estimator"><summary><pre class="estimator-name">PolynomialExtender</pre></summary><code class="estimator-params">
{'bias_name': 'bias',
 'degree': 2,
 'include_bias': False,
 'interaction_only': False}

</code></details><details class="component estimator"><summary><pre class="estimator-name">LinearRegression</pre></summary><code class="estimator-params">
{'_weights': {},
 '_y_name': None,
 'clip_gradient': 1000000000000.0,
 'initializer': Zeros (),
 'intercept': 0.0,
 'intercept_init': 0.0,
 'intercept_lr': Constant({'learning_rate': 0.01}),
 'l2': 0.0,
 'loss': Squared({}),
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



`compose.Pipeline` inherits from `base.Estimator`, which means that it has a `learn_one` method. You would expect `learn_one` to update each estimator, but **that's not actually what happens**. Instead, the transformers are updated when `predict_one` (or `predict_proba_one` for that matter) is called. Indeed, in online machine learning, we can update the unsupervised parts of our model when a sample arrives. We don't have to wait for the ground truth to arrive in order to update unsupervised estimators that don't depend on it. In other words, in a pipeline, `learn_one` updates the supervised parts, whilst `predict_one` updates the unsupervised parts. It's important to be aware of this behavior, as it is quite different to what is done in other libraries that rely on batch machine learning.

Here is a small example to illustrate the previous point:


```python
from river import datasets

dataset = datasets.TrumpApproval()
x, y = next(iter(dataset))
x, y
```




    ({'ordinal_date': 736389,
      'gallup': 43.843213,
      'ipsos': 46.19925042857143,
      'morning_consult': 48.318749,
      'rasmussen': 44.104692,
      'you_gov': 43.636914000000004},
     43.75505)



Let us call `predict_one`, which will update each transformer, but won't update the linear regression.


```python
model.predict_one(x)
```




    0.0



The prediction is nil because each weight of the linear regression is equal to 0.


```python
model['StandardScaler'].means
```




    defaultdict(float,
                {'ordinal_date': 736389.0,
                 'gallup': 43.843213,
                 'ipsos': 46.19925042857143,
                 'morning_consult': 48.318749,
                 'rasmussen': 44.104692,
                 'you_gov': 43.636914000000004})



As we can see, the means of each feature have been updated, even though we called `predict_one` and not `learn_one`.

Note that if you call `transform_one` with a pipeline who's last step is not a transformer, then the output from the last transformer (which is thus the penultimate step) will be returned:


```python
model.transform_one(x)
```




    {'ordinal_date': 0.0,
     'gallup': 0.0,
     'ipsos': 0.0,
     'morning_consult': 0.0,
     'rasmussen': 0.0,
     'you_gov': 0.0,
     'ordinal_date*ordinal_date': 0.0,
     'gallup*ordinal_date': 0.0,
     'ipsos*ordinal_date': 0.0,
     'morning_consult*ordinal_date': 0.0,
     'ordinal_date*rasmussen': 0.0,
     'ordinal_date*you_gov': 0.0,
     'gallup*gallup': 0.0,
     'gallup*ipsos': 0.0,
     'gallup*morning_consult': 0.0,
     'gallup*rasmussen': 0.0,
     'gallup*you_gov': 0.0,
     'ipsos*ipsos': 0.0,
     'ipsos*morning_consult': 0.0,
     'ipsos*rasmussen': 0.0,
     'ipsos*you_gov': 0.0,
     'morning_consult*morning_consult': 0.0,
     'morning_consult*rasmussen': 0.0,
     'morning_consult*you_gov': 0.0,
     'rasmussen*rasmussen': 0.0,
     'rasmussen*you_gov': 0.0,
     'you_gov*you_gov': 0.0}



In many cases, you might want to connect a step to multiple steps. For instance, you might to extract different kinds of features from a single input. An elegant way to do this is to use a `compose.TransformerUnion`. Essentially, the latter is a list of transformers who's results will be merged into a single `dict` when `transform_one` is called. As an example let's say that we want to apply a `feature_extraction.RBFSampler` as well as the `feature_extraction.PolynomialExtender`. This may be done as so:


```python
model = (
    preprocessing.StandardScaler() |
    (feature_extraction.PolynomialExtender() + feature_extraction.RBFSampler()) |
    linear_model.LinearRegression()
)

model
```




<div><div class="component pipeline"><details class="component estimator"><summary><pre class="estimator-name">StandardScaler</pre></summary><code class="estimator-params">
{'counts': Counter(),
 'means': defaultdict(&lt;class 'float'&gt;, {}),
 'vars': defaultdict(&lt;class 'float'&gt;, {}),
 'with_std': True}

</code></details><div class="component union"><details class="component estimator"><summary><pre class="estimator-name">PolynomialExtender</pre></summary><code class="estimator-params">
{'bias_name': 'bias',
 'degree': 2,
 'include_bias': False,
 'interaction_only': False}

</code></details><details class="component estimator"><summary><pre class="estimator-name">RBFSampler</pre></summary><code class="estimator-params">
{'gamma': 1.0,
 'n_components': 100,
 'offsets': [3.3510532411100926,
             1.190142184105075,
             4.758468173807059,
             2.102487972776319,
             1.480660275522741,
             5.366729269710237,
             5.070416334382951,
             1.277733738266996,
             2.207119719491707,
             2.1426794957848565,
             1.8225710193657765,
             0.053985537083313495,
             2.6438259461961584,
             0.8971767883543308,
             3.471297218403341,
             2.6459033776328047,
             1.9757793978738034,
             5.087466036691654,
             3.518448810009812,
             5.758687368535289,
             4.79849237290909,
             5.743603500595328,
             3.8577593701336594,
             2.992220690658145,
             4.6122186296260645,
             2.1072783802275836,
             3.2587620363834997,
             4.18188290647669,
             3.789865875963889,
             1.3166941816528979,
             2.1496598890995253,
             0.5514213256928427,
             6.133866289278633,
             5.464360858865711,
             4.291124096688779,
             5.030631537283815,
             0.9257361562479935,
             6.114310134092216,
             2.5412321682182526,
             4.822019847592126,
             0.49289853038336945,
             5.662322515846727,
             1.1066734350932208,
             2.7859787189161023,
             1.0923831484478823,
             3.862364264034545,
             4.57817015349273,
             4.879251535793154,
             3.2389588917501846,
             0.8671594208818371,
             0.3928381887147743,
             0.8367553042176593,
             5.487324628228967,
             1.754506781747617,
             1.4467895315220585,
             4.95057422570524,
             1.515646570633651,
             0.7141957541762096,
             2.6802165231715596,
             3.1143643491751765,
             3.2887707754133637,
             4.698780590052855,
             2.628523813897675,
             3.051845846553334,
             0.7137935763683448,
             0.8668790249131346,
             4.55066107913964,
             5.448264849218835,
             0.6859224016931418,
             3.7014814797697153,
             1.4690127755832323,
             4.680232230781756,
             2.1073009522234694,
             3.8794879419289177,
             0.9006462860416513,
             2.682015494386968,
             2.52991503710204,
             2.2535812651434344,
             5.407510051438392,
             2.8275014865565486,
             0.5645870178898392,
             5.242344945410286,
             1.609719168544174,
             4.340295048969955,
             2.927344299721854,
             2.1090856426673503,
             2.698017694795121,
             5.749312469665515,
             3.6999265064358733,
             3.280883125056919,
             5.481053451883993,
             1.77331580361824,
             0.0858416295646753,
             5.316833856722039,
             1.8426572018482168,
             5.347342560277715,
             5.1189679745176475,
             1.2592524362712572,
             3.637660524181257,
             2.393108047753942],
 'rng': &lt;random.Random object at 0x7ff61a344210&gt;,
 'seed': None,
 'weights': defaultdict(&lt;bound method RBFSampler._random_weights of RBFSampler (
  gamma=1.
  n_components=100
  seed=None
)&gt;, {})}

</code></details></div><details class="component estimator"><summary><pre class="estimator-name">LinearRegression</pre></summary><code class="estimator-params">
{'_weights': {},
 '_y_name': None,
 'clip_gradient': 1000000000000.0,
 'initializer': Zeros (),
 'intercept': 0.0,
 'intercept_init': 0.0,
 'intercept_lr': Constant({'learning_rate': 0.01}),
 'l2': 0.0,
 'loss': Squared({}),
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



Note that the `+` symbol acts as a shorthand notation for creating a `compose.TransformerUnion`, which means that we could have declared the above pipeline as so:


```python
model = (
    preprocessing.StandardScaler() |
    compose.TransformerUnion(
        feature_extraction.PolynomialExtender(),
        feature_extraction.RBFSampler()
    ) |
    linear_model.LinearRegression()
)
```

Pipelines provide the benefit of removing a lot of cruft by taking care of tedious details for you. They also enable to clearly define what steps your model is made of. Finally, having your model in a single object means that you can move it around more easily. Note that you can include user-defined functions in a pipeline by using a `compose.FuncTransformer`.
