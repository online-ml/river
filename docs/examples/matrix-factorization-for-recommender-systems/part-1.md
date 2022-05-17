# Part 1

**Table of contents of this tutorial series on matrix factorization for recommender systems:**

- [Part 1 - Traditional Matrix Factorization methods for Recommender Systems](/examples/matrix-factorization-for-recommender-systems-part-1)
- [Part 2 - Factorization Machines and Field-aware Factorization Machines](/examples/matrix-factorization-for-recommender-systems-part-2)
- [Part 3 - Large scale learning and better predictive power with multiple pass learning](/examples/matrix-factorization-for-recommender-systems-part-3)

## Introduction

A [recommender system](https://en.wikipedia.org/wiki/Recommender_system) is a software tool designed to generate and suggest items or entities to the users. Popular large scale examples include:

- Amazon (suggesting products)
- Facebook (suggesting posts in users' news feeds)
- Spotify (suggesting music)

Social recommendation from graph (mostly used by social networks) are not covered in `river`. We focus on the general case, item recommendation. This problem can be represented with the user-item matrix:

$$
\normalsize
\begin{matrix}
    & \begin{matrix} _1 & _\cdots & _\cdots & _\cdots & _I \end{matrix} \\
    \begin{matrix} _1 \\ _\vdots \\ _\vdots \\ _\vdots \\ _U \end{matrix} &
        \begin{bmatrix}
            {\color{Red} ?} & 2 & \cdots & {\color{Red} ?} & {\color{Red} ?} \\
            {\color{Red} ?} & {\color{Red} ?} & \cdots & {\color{Red} ?} & 4.5 \\
            \vdots & \ddots & \ddots & \ddots & \vdots \\
            3 & {\color{Red} ?} & \cdots & {\color{Red} ?} & {\color{Red} ?} \\
            {\color{Red} ?} & {\color{Red} ?} & \cdots & 5 & {\color{Red} ?}
        \end{bmatrix}
\end{matrix}
$$

Where $U$ and $I$ are the number of user and item of the system, respectively. A matrix entry represents a user's preference for an item, it can be a rating, a like or dislike, etc. Because of the huge number of users and items compared to the number of observed entries, those matrices are very sparsed (usually less than 1% filled).

[Matrix Factorization (MF)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) is a class of [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) algorithms derived from [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition). MF strength lies in its capacity to able to model high cardinality categorical variables interactions. This subfield boomed during the famous [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) contest in 2006, when numerous novel variants has been invented and became popular thanks to their attractive accuracy and scalability.

MF approach seeks to fill the user-item matrix considering the problem as a [matrix completion](https://en.wikipedia.org/wiki/Matrix_completion) one. MF core idea assume a latent model learning its own representation of the users and the items in a lower latent dimensional space by factorizing the observed parts of the matrix.

A factorized user or item is represented as a vector $\mathbf{v}_u$ or $\mathbf{v}_i$ composed of $k$ latent factors, with $k << U, I$. Those learnt latent variables represent, for an item the various aspects describing it, and for a user its interests in terms of those aspects. The model then assume a user's choice or fondness is composed of a sum of preferences about the various aspects of the concerned item. This sum being the dot product between the latent vectors of a given user-item pair:

$$
\normalsize
\langle \mathbf{v}_u, \mathbf{v}_i \rangle = \sum_{f=1}^{k} \mathbf{v}_{u, f} \cdot \mathbf{v}_{i, f}
$$

MF models weights are learnt in an online fashion, often with stochastic gradient descent as it provides relatively fast running time and good accuracy. There is a great and widely popular library named [surprise](http://surpriselib.com/) that implements MF models (and others) but in contrast with `river` doesn't follow a pure online philosophy (all the data have to be loaded in memory and the API doesn't allow you to update your model with new data).

**Notes:**

- In recent years, proposed deep learning techniques for recommendation tasks claim state of the art results. However, [recent work](https://arxiv.org/abs/1907.06902) (August 2019) showed that those promises can't be taken for granted and traditional MF methods are still relevant today.
- For more information about how the business value of recommender systems is measured and why they are one of the main success stories of machine learning, see the following [literature survey](https://arxiv.org/abs/1908.08328) (December 2019).

## Let's start

In this tutorial, we are going to explore MF algorithms available in `river` and test them on a movie recommendation problem with the MovieLens 100K dataset. This latter is a collection of movie ratings (from 1 to 5) that includes various information about both the items and the users. We can access it from the [river.datasets](/api/overview/#datasets) module:


```python
import json

from river import datasets

for x, y in datasets.MovieLens100K():
    print(f'x = {json.dumps(x, indent=4)}')
    print(f'y = {y}')
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


Let's define a routine to evaluate our different models on MovieLens 100K. Mean Absolute Error and Root Mean Squared Error will be our metrics printed alongside model's computation time and memory usage:


```python
from river import metrics
from river.evaluate import progressive_val_score

def evaluate(model, unpack_user_and_item=True):
    X_y = datasets.MovieLens100K(unpack_user_and_item)
    metric = metrics.MAE() + metrics.RMSE()
    _ = progressive_val_score(X_y, model, metric, print_every=25_000, show_time=True, show_memory=True)
```

## Naive prediction

It's good practice in machine learning to start with a naive baseline and then iterate from simple things to complex ones observing progress incrementally. Let's start by predicting the target running mean as a first shot:


```python
from river import dummy
from river import stats

model = dummy.StatisticRegressor(stats.Mean())
evaluate(model, unpack_user_and_item=False)
```

    [25,000] MAE: 0.934259, RMSE: 1.124469 – 00:00:00 – 514 B
    [50,000] MAE: 0.923893, RMSE: 1.105 – 00:00:01 – 514 B
    [75,000] MAE: 0.937359, RMSE: 1.123696 – 00:00:01 – 514 B
    [100,000] MAE: 0.942162, RMSE: 1.125783 – 00:00:02 – 514 B


## Baseline model

Now we can do machine learning and explore available models in [river.reco](https://online-ml.github.io/api/overview/#reco) module starting with the baseline model. It extends our naive prediction by adding to the global running mean two bias terms characterizing the user and the item discrepancy from the general tendency. The model equation is defined as:

$$
\normalsize
\hat{y}(x) = \bar{y} + bu_{u} + bi_{i}
$$

This baseline model can be viewed as a linear regression where the intercept is replaced by the target running mean with the users and the items one hot encoded.

All machine learning models in `river` expect dicts as input with feature names as keys and feature values as values. Specifically, models from `river.reco` expect a `'user'` and an `'item'` entries without any type constraint on their values (i.e. can be strings or numbers), e.g.:

```python
x = {
    'user': 'Guido',
    'item': "Monty Python's Flying Circus"
}
```

Other entries, if exist, are simply ignored. This is quite useful as we don't need to spend time and storage doing one hot encoding.


```python
from river import preprocessing
from river import optim
from river import reco

baseline_params = {
    'optimizer': optim.SGD(0.025),
    'l2': 0.,
    'initializer': optim.initializers.Zeros()
}

model = preprocessing.PredClipper(
    regressor=reco.Baseline(**baseline_params),
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: 0.761844, RMSE: 0.960972 – 0:00:00.864336 – 132.26 KB
    [50,000] MAE: 0.753292, RMSE: 0.951223 – 0:00:01.737809 – 191.78 KB
    [75,000] MAE: 0.754177, RMSE: 0.953376 – 0:00:02.598330 – 225.88 KB
    [100,000] MAE: 0.754651, RMSE: 0.954148 – 0:00:03.464756 – 240.29 KB


We won two tenth of MAE compared to our naive prediction (0.7546 vs 0.9421) meaning that significant information has been learnt by the model.

## Funk Matrix Factorization (FunkMF)

It's the pure form of matrix factorization consisting of only learning the users and items latent representations as discussed in introduction. Simon Funk popularized its [stochastic gradient descent optimization](https://sifter.org/simon/journal/20061211.html) in 2006 during the Netflix Prize. The model equation is defined as:

$$
\normalsize
\hat{y}(x) = \langle \mathbf{v}_u, \mathbf{v}_i \rangle
$$

**Note:** FunkMF is sometimes referred as [Probabilistic Matrix Factorization](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf) which is an extended probabilistic version.


```python
funk_mf_params = {
    'n_factors': 10,
    'optimizer': optim.SGD(0.05),
    'l2': 0.1,
    'initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73)
}

model = preprocessing.PredClipper(
    regressor=reco.FunkMF(**funk_mf_params),
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: 1.070136, RMSE: 1.397014 – 0:00:01.705144 – 938.07 KB
    [50,000] MAE: 0.99174, RMSE: 1.290666 – 0:00:03.466905 – 1.13 MB
    [75,000] MAE: 0.961072, RMSE: 1.250842 – 0:00:05.205363 – 1.33 MB
    [100,000] MAE: 0.944883, RMSE: 1.227688 – 0:00:06.934770 – 1.5 MB


Results are equivalent to our naive prediction (0.9448 vs 0.9421). By only focusing on the users preferences and the items characteristics, the model is limited in his ability to capture different views of the problem. Despite its poor performance alone, this algorithm is quite useful combined in other models or when we need to build dense representations for other tasks.

## Biased Matrix Factorization (BiasedMF)

It's the combination of the Baseline model and FunkMF. The model equation is defined as:

$$
\normalsize
\hat{y}(x) = \bar{y} + bu_{u} + bi_{i} + \langle \mathbf{v}_u, \mathbf{v}_i \rangle
$$

**Note:** *Biased Matrix Factorization* name is used by some people but some others refer to it by *SVD* or *Funk SVD*. It's the case of Yehuda Koren and Robert Bell in [Recommender Systems Handbook](https://www.cse.iitk.ac.in/users/nsrivast/HCC/Recommender_systems_handbook.pdf) (Chapter 5 *Advances in Collaborative Filtering*) and of `surprise` library. Nevertheless, *SVD* could be confused with the original *Singular Value Decomposition* from which it's derived from, and *Funk SVD* could also be misleading because of the biased part of the model equation which doesn't come from Simon Funk's work. For those reasons, we chose to side with *Biased Matrix Factorization* which fits more naturally to it.


```python
biased_mf_params = {
    'n_factors': 10,
    'bias_optimizer': optim.SGD(0.025),
    'latent_optimizer': optim.SGD(0.05),
    'weight_initializer': optim.initializers.Zeros(),
    'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73),
    'l2_bias': 0.,
    'l2_latent': 0.
}

model = preprocessing.PredClipper(
    regressor=reco.BiasedMF(**biased_mf_params),
    y_min=1,
    y_max=5
)

evaluate(model)
```

    [25,000] MAE: 0.761818, RMSE: 0.961057 – 0:00:01.917323 – 1.01 MB
    [50,000] MAE: 0.751667, RMSE: 0.949443 – 0:00:03.825794 – 1.28 MB
    [75,000] MAE: 0.749653, RMSE: 0.948723 – 0:00:05.737369 – 1.51 MB
    [100,000] MAE: 0.748559, RMSE: 0.947854 – 0:00:07.666314 – 1.69 MB


Results improved (0.7485 vs 0.7546) demonstrating that users and items latent representations bring additional information.

To conclude this first tutorial about factorization models, let's review the important parameters to tune when dealing with this family of methods:

- `n_factors`: the number of latent factors. The more you set, the more items aspects and users preferences you are going to learn. Too many will cause overfitting, `l2` regularization could help.
- `*_optimizer`: the optimizers. Classic stochastic gradient descent performs well, finding the good learning rate will make the difference.
- `initializer`: the latent weights initialization. Latent vectors have to be initialized with non-constant values. We generally sample them from a zero-mean normal distribution with small standard deviation.
