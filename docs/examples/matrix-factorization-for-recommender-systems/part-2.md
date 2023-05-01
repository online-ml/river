# Part 2


As seen in [Part 1](/examples/matrix-factorization-for-recommender-systems-part-1), strength of [Matrix Factorization (MF)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) lies in its ability to deal with sparse and high cardinality categorical variables. In this second tutorial we will have a look at Factorization Machines (FM) algorithm and study how it generalizes the power of MF.

**Table of contents of this tutorial series on matrix factorization for recommender systems:**

- [Part 1 - Traditional Matrix Factorization methods for Recommender Systems](https://online-ml.github.io/examples/matrix-factorization-for-recommender-systems-part-1)
- [Part 2 - Factorization Machines and Field-aware Factorization Machines](https://online-ml.github.io/examples/matrix-factorization-for-recommender-systems-part-2)
- [Part 3 - Large scale learning and better predictive power with multiple pass learning](https://online-ml.github.io/examples/matrix-factorization-for-recommender-systems-part-3)

## Factorization Machines

Steffen Rendel came up in 2010 with [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf), an algorithm able to handle any real valued feature vector, combining the advantages of general predictors with factorization models. It became quite popular in the field of online advertising, notably after winning several Kaggle competitions. The modeling technique starts with a linear regression to capture the effects of each variable individually:

$$
\normalsize
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}
$$

Then are added interaction terms to learn features relations. Instead of learning a single and specific weight per interaction (as in [polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression)), a set of latent factors is learnt per feature (as in MF). An interaction is calculated by multiplying involved features product with their latent vectors dot product. The degree of factorization — or model order — represents the maximum number of features per interaction considered. The model equation for a factorization machine of degree $d$ = 2 is defined as:

$$
\normalsize
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}  + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_{j} x_{j'}
$$

Where $\normalsize \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle$ is the dot product of $j$ and $j'$ latent vectors:

$$
\normalsize
\langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle = \sum_{f=1}^{k} \mathbf{v}_{j, f} \cdot \mathbf{v}_{j', f}
$$

Higher-order FM will be covered in a following section, just note that factorization models express their power in sparse settings, which is also where higher-order interactions are hard to estimate.

Strong emphasis must be placed on feature engineering as it allows FM to mimic most factorization models and significantly impact its performance. High cardinality categorical variables one hot encoding is the most frequent step before feeding the model with data. For more efficiency, River FM implementation considers string values as categorical variables and automatically one hot encode them. FM models have their own module [river.facto](/api/overview/#facto).

 ## Mimic Biased Matrix Factorization (BiasedMF)

Let's start with a simple example where we want to reproduce the Biased Matrix Factorization model we trained in the previous tutorial. For a fair comparison with [Part 1 example](/examples/matrix-factorization-for-recommender-systems-part-1/#biased-matrix-factorization-biasedmf), let's set the same evaluation framework:


```python
from river import datasets
from river import metrics
from river.evaluate import progressive_val_score

def evaluate(model):
    X_y = datasets.MovieLens100K()
    metric = metrics.MAE() + metrics.RMSE()
    _ = progressive_val_score(X_y, model, metric, print_every=25_000, show_time=True, show_memory=True)
```

In order to build an equivalent model we need to use the same hyper-parameters. As we can't replace FM intercept by the global running mean we won't be able to build the exact same model:


```python
from river import compose
from river import facto
from river import preprocessing
from river import optim
from river import stats

fm_params = {
    'n_factors': 10,
    'weight_optimizer': optim.SGD(0.025),
    'latent_optimizer': optim.SGD(0.05),
    'sample_normalization': False,
    'l1_weight': 0.,
    'l2_weight': 0.,
    'l1_latent': 0.,
    'l2_latent': 0.,
    'intercept': 3,
    'intercept_lr': .01,
    'weight_initializer': optim.initializers.Zeros(),
    'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73),
}

regressor = compose.Select('user', 'item')
regressor |= facto.FMRegressor(**fm_params)

model = preprocessing.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: nan, RMSE: nan – 00:00:04 – 6.55 KB
    [50,000] MAE: nan, RMSE: nan – 00:00:08 – 6.55 KB
    [75,000] MAE: nan, RMSE: nan – 00:00:12 – 6.55 KB
    [100,000] MAE: nan, RMSE: nan – 00:00:16 – 6.55 KB


Both MAE are very close to each other (0.7486 vs 0.7485) showing that we almost reproduced [reco.BiasedMF](/api/reco/BiasedMF/) algorithm. The cost is a naturally slower running time as FM implementation offers more flexibility.

## Feature engineering for FM models

Let's study the basics of how to properly encode data for FM models. We are going to keep using MovieLens 100K as it provides various feature types:


```python
import json

for x, y in datasets.MovieLens100K():
    print(f'x = {json.dumps(x, indent=4)}\ny = {y}')
    break
```

    x = {
        "user": 259,
        "item": 255,
        "timestamp": 874731910000000000,
        "title": "My Best Friend's Wedding (1997)",
        "release_date": 866764800000000000,
        "genres": "comedy, romance",
        "age": 21.0,
        "gender": "M",
        "occupation": "student",
        "zip_code": "48823"
    }
    y = 4.0


The features we are going to add to our model don't improve its predictive power. Nevertheless, they are useful to illustrate different methods of data encoding:

1. Set-categorical variables

We have seen that categorical variables are one hot encoded automatically if set to strings, in the other hand, set-categorical variables must be encoded explicitly by the user. A good way of doing so is to assign them a value of $1/m$, where $m$ is the number of elements of the sample set. It gives the feature a constant "weight" across all samples preserving model's stability. Let's create a routine to encode movies genres this way:


```python
def split_genres(x):
    genres = x['genres'].split(', ')
    return {f'genre_{genre}': 1 / len(genres) for genre in genres}
```

2. Numerical variables

In practice, transforming numerical features into categorical ones works better in most cases. Feature binning is the natural way, but finding good bins is sometimes more an art than a science. Let's encode users age with something simple:


```python
def bin_age(x):
    if x['age'] <= 18:
        return {'age_0-18': 1}
    elif x['age'] <= 32:
        return {'age_19-32': 1}
    elif x['age'] < 55:
        return {'age_33-54': 1}
    else:
        return {'age_55-100': 1}
```

Let's put everything together:


```python
fm_params = {
    'n_factors': 14,
    'weight_optimizer': optim.SGD(0.01),
    'latent_optimizer': optim.SGD(0.025),
    'intercept': 3,
    'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.05, seed=73),
}

regressor = compose.Select('user', 'item')
regressor += (
    compose.Select('genres') |
    compose.FuncTransformer(split_genres)
)
regressor += (
    compose.Select('age') |
    compose.FuncTransformer(bin_age)
)
regressor |= facto.FMRegressor(**fm_params)

model = preprocessing.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: nan, RMSE: nan – 00:00:10 – 30.31 KB



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /var/folders/9z/dgt2y49d2qx_hkgt7qj8mc240000gn/T/ipykernel_36090/4267340742.py in <module>
         24 )
         25
    ---> 26 evaluate(model)


    /var/folders/9z/dgt2y49d2qx_hkgt7qj8mc240000gn/T/ipykernel_36090/2127757032.py in evaluate(model)
          6     X_y = datasets.MovieLens100K(unpack_user_and_item=False)
          7     metric = metrics.MAE() + metrics.RMSE()
    ----> 8     _ = progressive_val_score(X_y, model, metric, print_every=25_000, show_time=True, show_memory=True)


    ~/projects/river/river/evaluate/progressive_validation.py in progressive_val_score(dataset, model, metric, moment, delay, print_every, show_time, show_memory, **print_kwargs)
        238     )
        239
    --> 240     for checkpoint in checkpoints:
        241
        242         msg = f"[{checkpoint['Step']:,d}] {metric}"


    ~/projects/river/river/evaluate/progressive_validation.py in _progressive_validation(dataset, model, metric, checkpoints, moment, delay, measure_time, measure_memory)
         55             metric.update(y_true=y, y_pred=y_pred)
         56         if model._supervised:
    ---> 57             model.learn_one(x=x, y=y, **kwargs)
         58         else:
         59             model.learn_one(x=x, **kwargs)


    ~/projects/river/river/preprocessing/pred_clipper.py in learn_one(self, x, y, **kwargs)
         54
         55     def learn_one(self, x, y, **kwargs):
    ---> 56         self.regressor.learn_one(x=x, y=y, **kwargs)
         57         return self
         58


    ~/projects/river/river/compose/pipeline.py in learn_one(self, x, y, **params)
        502         last_step = next(steps)
        503         if last_step._supervised:
    --> 504             last_step.learn_one(x=x, y=y, **params)
        505         else:
        506             last_step.learn_one(x, **params)


    ~/projects/river/river/facto/base.py in learn_one(self, x, y, sample_weight)
         75             x = {j: xj / x_l2_norm for j, xj in x.items()}
         76
    ---> 77         return self._learn_one(x, y, sample_weight=sample_weight)
         78
         79     def _ohe_cat_features(self, x):


    ~/projects/river/river/facto/base.py in _learn_one(self, x, y, sample_weight)
        105
        106         # Update the latent weights
    --> 107         self._update_latents(x, g_loss)
        108
        109         return self


    ~/projects/river/river/facto/fm.py in _update_latents(self, x, g_loss)
         91         gradients = {}
         92         for j, xj in x.items():
    ---> 93             gradients[j] = {
         94                 f: g_loss * (xj * precomputed_sum[f] - v[j][f] * xj ** 2)
         95                 + l1 * sign(v[j][f])


    ~/projects/river/river/facto/fm.py in <dictcomp>(.0)
         91         gradients = {}
         92         for j, xj in x.items():
    ---> 93             gradients[j] = {
         94                 f: g_loss * (xj * precomputed_sum[f] - v[j][f] * xj ** 2)
         95                 + l1 * sign(v[j][f])


    KeyboardInterrupt:


Note that using more variables involves factorizing a larger latent space, then increasing the number of latent factors $k$ often helps capturing more information.

Some other feature engineering tips from [3 idiots' winning solution](https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10555) for Kaggle [Criteo display ads](https://www.kaggle.com/c/criteo-display-ad-challenge) competition in 2014:

- Infrequent modalities often bring noise and little information, transforming them into a special tag can help
- In some cases, sample-wise normalization seems to make the optimization problem easier to be solved

## Higher-Order Factorization Machines (HOFM)

The model equation generalized to any order $d \geq 2$ is defined as:

$$
\normalsize
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}  + \sum_{l=2}^{d} \sum_{j_1=1}^{p} \cdots \sum_{j_l=j_{l-1}+1}^{p} \left(\prod_{j'=1}^{l} x_{j_{j'}} \right) \left(\sum_{f=1}^{k_l} \prod_{j'=1}^{l} v_{j_{j'}, f}^{(l)} \right)
$$


```python
hofm_params = {
    'degree': 3,
    'n_factors': 12,
    'weight_optimizer': optim.SGD(0.01),
    'latent_optimizer': optim.SGD(0.025),
    'intercept': 3,
    'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.05, seed=73),
}

regressor = compose.Select('user', 'item')
regressor += (
    compose.Select('genres') |
    compose.FuncTransformer(split_genres)
)
regressor += (
    compose.Select('age') |
    compose.FuncTransformer(bin_age)
)
regressor |= facto.HOFMRegressor(**hofm_params)

model = preprocessing.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: 0.761297, RMSE: 0.962054 – 0:00:51.632190 – 2.61 MB
    [50,000] MAE: 0.751865, RMSE: 0.951499 – 0:01:42.890329 – 3.08 MB
    [75,000] MAE: 0.750853, RMSE: 0.951526 – 0:02:34.207244 – 3.6 MB
    [100,000] MAE: 0.750607, RMSE: 0.951982 – 0:03:25.248686 – 4.07 MB


As said previously, high-order interactions are often hard to estimate due to too much sparsity, that's why we won't spend too much time here.

## Field-aware Factorization Machines (FFM)

[Field-aware variant of FM (FFM)](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) improved the original method by adding the notion of "*fields*". A "*field*" is a group of features that belong to a specific domain (e.g. the "*users*" field, the "*items*" field, or the "*movie genres*" field).

FFM restricts itself to pairwise interactions and factorizes separated latent spaces — one per combination of fields (e.g. users/items, users/movie genres, or items/movie genres) — instead of a common one shared by all fields. Therefore, each feature has one latent vector per field it can interact with — so that it can learn the specific effect with each different field.

The model equation is defined by:

$$
\normalsize
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}  + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_{j, f_{j'}}, \mathbf{v}_{j', f_{j}} \rangle x_{j} x_{j'}
$$

Where $f_j$ and $f_{j'}$ are the fields corresponding to $j$ and $j'$ features, respectively.


```python
ffm_params = {
    'n_factors': 8,
    'weight_optimizer': optim.SGD(0.01),
    'latent_optimizer': optim.SGD(0.025),
    'intercept': 3,
    'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.05, seed=73),
}

regressor = compose.Select('user', 'item')
regressor += (
    compose.Select('genres') |
    compose.FuncTransformer(split_genres)
)
regressor += (
    compose.Select('age') |
    compose.FuncTransformer(bin_age)
)
regressor |= facto.FFMRegressor(**ffm_params)

model = preprocessing.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: 0.757718, RMSE: 0.958158 – 0:00:15.781740 – 3.04 MB
    [50,000] MAE: 0.749502, RMSE: 0.948065 – 0:00:31.431484 – 3.59 MB
    [75,000] MAE: 0.749275, RMSE: 0.948918 – 0:00:47.079510 – 4.19 MB
    [100,000] MAE: 0.749542, RMSE: 0.949769 – 0:01:02.776969 – 4.75 MB


Note that FFM usually needs to learn smaller number of latent factors $k$ than FM as each latent vector only deals with one field.

## Field-weighted Factorization Machines (FwFM)

[Field-weighted Factorization Machines (FwFM)](https://arxiv.org/abs/1806.03514) address FFM memory issues caused by its large number of parameters, which is in the order of *feature number* times *field number*. As FFM, FwFM is an extension of FM restricted to pairwise interactions, but instead of factorizing separated latent spaces, it learns a specific weight $r_{f_j, f_{j'}}$ for each field combination modelling the interaction strength.

The model equation is defined as:

$$
\normalsize
\hat{y}(x) = w_{0} + \sum_{j=1}^{p} w_{j} x_{j}  + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} r_{f_j, f_{j'}} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_{j} x_{j'}
$$


```python
fwfm_params = {
    'n_factors': 10,
    'weight_optimizer': optim.SGD(0.01),
    'latent_optimizer': optim.SGD(0.025),
    'intercept': 3,
    'seed': 73,
}

regressor = compose.Select('user', 'item')
regressor += (
    compose.Select('genres') |
    compose.FuncTransformer(split_genres)
)
regressor += (
    compose.Select('age') |
    compose.FuncTransformer(bin_age)
)
regressor |= facto.FwFMRegressor(**fwfm_params)

model = preprocessing.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: 0.761539, RMSE: 0.962241 – 0:00:20.963815 – 1.18 MB
    [50,000] MAE: 0.754089, RMSE: 0.953181 – 0:00:42.057991 – 1.38 MB
    [75,000] MAE: 0.754806, RMSE: 0.954979 – 0:01:04.051777 – 1.6 MB
    [100,000] MAE: 0.755404, RMSE: 0.95604 – 0:01:25.823651 – 1.79 MB

