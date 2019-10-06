<div align="center">
  <img height="240px" src="https://docs.google.com/drawings/d/e/2PACX-1vSl80T4MnWRsPX3KvlB2kn6zVdHdUleG_w2zBiLS7RxLGAHxiSYTnw3LZtXh__YMv6KcIOYOvkSt9PB/pub?w=841&h=350" alt="creme_logo"/>
</div>

<div align="center">
  <!-- Travis -->
  <a href="https://travis-ci.org/creme-ml/creme">
    <img src="https://img.shields.io/travis/creme-ml/creme/master.svg?style=for-the-badge" alt="travis" />
  </a>
  <!-- Codecov -->
  <a href="https://codecov.io/gh/creme-ml/creme">
    <img src="https://img.shields.io/codecov/c/gh/creme-ml/creme.svg?style=for-the-badge" alt="codecov" />
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/creme">
    <img src="https://img.shields.io/pypi/v/creme.svg?style=for-the-badge" alt="pypi" />
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=for-the-badge" alt="bsd_3_license"/>
  </a>
</div>

<br/>

`creme` is a library for online machine learning, also known as in**creme**ntal learning. Online learning is a machine learning regime where a **model learns one observation at a time**. This is in contrast to batch learning where all the data is processed in one go. Incremental learning is desirable when the data is too big to fit in memory, or simply when you want to **handle streaming data**. In addition to many online machine learning algorithms, `creme` provides utilities for **extracting features from a stream of data**.

## Useful links

- [Documentation](https://creme-ml.github.io/)
  - [API reference](https://creme-ml.github.io/api.html)
  - [User guide](https://creme-ml.github.io/user-guide.html)
  - [FAQ](https://creme-ml.github.io/faq.html)
- [Benchmarks](https://github.com/creme-ml/creme/tree/master/benchmarks)
- [Issue tracker](https://github.com/creme-ml/creme/issues)
- [Package releases](https://pypi.org/project/creme/#history)
- [Change history](https://github.com/creme-ml/creme/blob/master/CHANGELOG.md)
- PyData Amsterdam 2019 presentation ([slides](https://maxhalford.github.io/slides/creme-pydata/), [video](https://www.youtube.com/watch?v=P3M6dt7bY9U&list=PLGVZCDnMOq0q7_6SdrC2wRtdkojGBTAht&index=11))
- [Toulouse Data Science presentation](https://maxhalford.github.io/slides/creme-tds/)

## Installation

:point_up: `creme` is intended to work with Python 3.6 and above.

`creme` can simply be installed with `pip`.

    pip install creme

You can also install the bleeding edge version as so:

    pip install git+https://github.com/creme-ml/creme
    # Or through SSH:
    pip install git+ssh://git@github.com/creme-ml/creme.git

If you're looking to contribute to ``creme`` and want to have a development setup, then please check out the [contribution guidelines](CONTRIBUTING.md).

## Example

In the following example we'll use a linear regression to forecast the number of available bikes in [bike stations](https://www.wikiwand.com/en/Bicycle-sharing_system) from the city of Toulouse. Each observation looks like this:

```python
>>> import pprint
>>> from creme import datasets

>>> X_y = datasets.fetch_bikes()
>>> x, y = next(X_y)

>>> pprint.pprint(x)
{'clouds': 75,
 'description': 'light rain',
 'humidity': 81,
 'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
 'pressure': 1017.0,
 'station': 'metro-canal-du-midi',
 'temperature': 6.54,
 'wind': 9.3}

>>> print(f'Number of bikes: {y}')
Number of bikes: 1

```

We will include all the available numeric features in our model. We will also use target encoding by calculating a running average of the target per station and hour. Before being fed to the linear regression, the features will be scaled using a `StandardScaler`. Note that each of these steps works in a streaming fashion, including the feature extraction. We'll evaluate the model by asking it to forecast 30 minutes ahead while delaying the true answers, which ensures that we're simulating a production scenario. Finally we will print the current score every 20,000 predictions.

```python
>>> import datetime as dt
>>> from creme import compose
>>> from creme import datasets
>>> from creme import feature_extraction
>>> from creme import linear_model
>>> from creme import metrics
>>> from creme import model_selection
>>> from creme import preprocessing
>>> from creme import stats

>>> X_y = datasets.fetch_bikes()

>>> def add_hour(x):
...     x['hour'] = x['moment'].hour
...     return x

>>> model = compose.Whitelister('clouds', 'humidity', 'pressure', 'temperature', 'wind')
>>> model += (
...     add_hour |
...     feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
... )
>>> model += feature_extraction.TargetAgg(by='station', how=stats.EWMean(0.5))
>>> model |= preprocessing.StandardScaler()
>>> model |= linear_model.LinearRegression()

>>> model_selection.online_qa_score(
...     X_y=X_y,
...     model=model,
...     metric=metrics.MAE(),
...     on='moment',
...     lag=dt.timedelta(minutes=30),
...     print_every=30_000
... )
[30,000] MAE: 2.193069
[60,000] MAE: 2.249345
[90,000] MAE: 2.288321
[120,000] MAE: 2.265257
[150,000] MAE: 2.2674
[180,000] MAE: 2.282485
MAE: 2.285921

```

You can visualize the pipeline as so:

```python
>>> model
Pipeline (
    TransformerUnion (
        Whitelister (
            whitelist=['clouds', 'humidity', 'pressure', 'temperature', 'wind']
        ),
        Pipeline (
            FuncTransformer (
                func=add_hour
            ),
            TargetAgg (
                by=['station', 'hour']
                how=Mean: 0.
                target_name='target'
            )
        ),
        TargetAgg (
            by=['station']
            how=EWMean: 0.
            target_name='target'
        )
    ),
    StandardScaler (),
    LinearRegression (
        optimizer=SGD
        loss=Squared
        l2=0.0001
        intercept=0.0
        intercept_lr=0.01
    )
)

```

We can also draw the pipeline.

```python
>>> dot = model.draw()

```

<div align="center">
  <img src="./docs/_static/bikes_pipeline.svg" alt="bikes_pipeline"/>
</div>

By only using a few lines of code, we've built a robust model and evaluated it by simulating a production scenario. You can find a more detailed version of this example [here](https://creme-ml.github.io/notebooks/bike-sharing-forecasting.html). `creme` is a framework that has a lot to offer, and as such we kindly refer you to the [documentation](https://creme-ml.github.io/) if you want to know more.

## Contributing

Like many subfields of machine learning, online learning is far from being an exact science and so there is still a lot to do. Feel free to contribute in any way you like, we're always open to new ideas and approaches. If you want to contribute to the code base please check out the [CONTRIBUTING.md file](https://github.com/creme-ml/creme/blob/master/CONTRIBUTING.md). Also take a look at the [issue tracker](https://github.com/creme-ml/creme/issues) and see if anything takes your fancy.

Last but not least you are more than welcome to share with us on how you're using `creme` or online learning in general! We believe that online learning solves a lot of pain points in practice, and would love to share experiences.

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://maxhalford.github.io"><img src="https://avatars1.githubusercontent.com/u/8095957?v=4" width="100px;" alt="Max Halford"/><br /><sub><b>Max Halford</b></sub></a><br /><a href="#projectManagement-MaxHalford" title="Project Management">📆</a> <a href="https://github.com/Max Halford/creme/commits?author=MaxHalford" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/AdilZouitine"><img src="https://avatars0.githubusercontent.com/u/24889239?v=4" width="100px;" alt="AdilZouitine"/><br /><sub><b>AdilZouitine</b></sub></a><br /><a href="https://github.com/Max Halford/creme/commits?author=AdilZouitine" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/raphaelsty"><img src="https://avatars3.githubusercontent.com/u/24591024?v=4" width="100px;" alt="Raphael Sourty"/><br /><sub><b>Raphael Sourty</b></sub></a><br /><a href="https://github.com/Max Halford/creme/commits?author=raphaelsty" title="Code">💻</a></td>
    <td align="center"><a href="http://www.linkedin.com/in/gbolmier"><img src="https://avatars0.githubusercontent.com/u/25319692?v=4" width="100px;" alt="Geoffrey Bolmier"/><br /><sub><b>Geoffrey Bolmier</b></sub></a><br /><a href="https://github.com/Max Halford/creme/commits?author=gbolmier" title="Code">💻</a></td>
    <td align="center"><a href="http://koaning.io"><img src="https://avatars1.githubusercontent.com/u/1019791?v=4" width="100px;" alt="vincent d warmerdam "/><br /><sub><b>vincent d warmerdam </b></sub></a><br /><a href="https://github.com/Max Halford/creme/commits?author=koaning" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/VaysseRobin"><img src="https://avatars2.githubusercontent.com/u/32324822?v=4" width="100px;" alt="VaysseRobin"/><br /><sub><b>VaysseRobin</b></sub></a><br /><a href="https://github.com/Max Halford/creme/commits?author=VaysseRobin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/tweakyllama"><img src="https://avatars0.githubusercontent.com/u/7049400?v=4" width="100px;" alt="Lygon Bowen-West"/><br /><sub><b>Lygon Bowen-West</b></sub></a><br /><a href="https://github.com/Max Halford/creme/commits?author=tweakyllama" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/flegac"><img src="https://avatars2.githubusercontent.com/u/4342302?v=4" width="100px;" alt="Florent Le Gac"/><br /><sub><b>Florent Le Gac</b></sub></a><br /><a href="https://github.com/Max Halford/creme/commits?author=flegac" title="Code">💻</a></td>
    <td align="center"><a href="http://www.pyimagesearch.com"><img src="https://avatars2.githubusercontent.com/u/759645?v=4" width="100px;" alt="Adrian Rosebrock"/><br /><sub><b>Adrian Rosebrock</b></sub></a><br /><a href="#blog-jrosebr1" title="Blogposts">📝</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

## License

See the [license file](LICENSE).
