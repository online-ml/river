import copy
import math

from .. import base

from . import grid


__all__ = ['successive_halving']


def successive_halving(model, param_grid, X_y, metric, budget, eta=2, verbose=True):
    """Successive halving algorithm.

    Successive halving is a method for performing model selection without having to train each
    model on all the dataset. At certain points in time (called "rungs"), the worst performing will
    be discarded and the best ones will keep competing between each other. The rung values are
    designed so that at most ``budget`` model updates will be performed in total.

    If you have ``k`` combinations of hyperparameters and that your dataset contains ``n``
    observations, then the maximal budget you can allocate is:

    .. math:: \\frac{2kn}{eta}

    It is recommended that you check this beforehand. This bound can't be checked by the function
    because the size of the dataset is not known. In fact it is potentially infinite, in which case
    the algorithm will terminate once all the budget has been spent.

    If you have a budget of ``B``, and that your dataset contains ``n`` observations, then the
    number of hyperparameter combinations that will spend all the budget and go through all the
    data is:

    .. math:: \\ceil(\\floor(B / (n * 2)) * eta)

    Parameters:
        model (models.Estimator)
        param_grid (dict): Parameter grid.
        X_y (generator): A stream of (features, target) tuples.
        metric (metrics.Metric): Metric used for comparing models with.
        budget (int): Total number of model updates you wish to allocate.
        eta (int): Rate of elimination. At every rung, ``math.ceil(k / eta)`` models are kept,
            where ``k`` is the number of models that have reached the rung. A higher ``eta`` value
            will focus on less models but will allocate more iterations to the best models.
        verbose (bool): Whether to display progress or not.

    Returns:
        dict: The best configuration. You can use ``model._reset(params)`` to use these parameters.

    Examples:

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import preprocessing
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_boston(),
            ...     shuffle=True,
            ...     random_state=42
            ... )

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LinearRegression()
            ... )

            >>> param_grid = {
            ...     'LinearRegression': {
            ...         'optimizer': [
            ...             (optim.SGD, {'lr': [1, 2, 3]}),
            ...             (optim.Adam, {'beta_1': [0.01, 0.001], 'lr': [0.1, 0.01, 0.001]}),
            ...             (optim.Adam, {'beta_1': [0.1], 'lr': [0.001]}),
            ...         ]
            ...     }
            ... }

            >>> best_params = model_selection.successive_halving(
            ...     model=model,
            ...     param_grid=param_grid,
            ...     X_y=X_y,
            ...     metric=metrics.MSE(),
            ...     budget=2000,
            ...     eta=2
            ... )
            ... # doctest: +NORMALIZE_WHITESPACE
            [1] 5 removed   5 left  50 iterations   budget used: 500    budget left: 1500   best MSE: 254.450992
            [2] 2 removed   3 left  100 iterations  budget used: 1000   budget left: 1000   best MSE: 169.045235
            [3] 1 removed   2 left  166 iterations  budget used: 1498   budget left: 502    best MSE: 113.623094
            [4] 1 removed   1 left  250 iterations  budget used: 1998   budget left: 2      best MSE: 90.453983

            >>> best_model = model._set_params(**best_params)
            >>> best_model
            Pipeline (
              StandardScaler (),
              LinearRegression (
                optimizer=Adam (
                  lr=Constant (
                    learning_rate=0.1
                  )
                  beta_1=0.001
                  beta_2=0.999
                  eps=1e-08
                )
                loss=Squared ()
                l2=0.
                intercept=0.
                intercept_lr=Constant (
                  learning_rate=0.01
                )
                clip_gradient=1e+12
              )
            )

    References:
        1. `Non-stochastic Best Arm Identification and Hyperparameter Optimization <http://proceedings.mlr.press/v51/jamieson16.pdf>`_
        2. `Massively Parallel Hyperparameter Tuning <https://arxiv.org/pdf/1810.05934.pdf>`_
        3. `Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization <https://arxiv.org/pdf/1603.06560.pdf>`_

    """

    def _print(msg):
        if verbose:
            print(msg)

    # Generate models
    param_grid = list(grid.expand_param_grid(param_grid))
    models = [model._set_params(**params) for params in param_grid]

    # Check that the model and the metric are in accordance
    for model in models:
        if not metric.works_with(model):
            raise ValueError(f"{metric.__class__.__name__} metric can't be used to evaluate a " +
                             f'{model.__class__.__name__}')

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    pred_func = lambda model: model.predict_one
    if isinstance(model, base.Classifier) and not metric.requires_labels:
        pred_func = lambda model: model.predict_proba_one

    # Instantiate one metric per model
    n = len(models)
    metrics = [copy.copy(metric) for _ in range(n)]
    n_rungs = 0

    rankings = list(range(n))
    s = n
    r = math.floor(budget / (s * math.ceil(math.log(n, eta))))
    budget_used = 0

    for i, (x, y) in enumerate(X_y, start=1):

        # Loop over the s best models
        for j in rankings[:s]:
            model = models[j]
            metric = metrics[j]
            y_pred = pred_func(model)(x)
            metric.update(y_true=y, y_pred=y_pred)
            model.fit_one(x, y)

        if i == r:

            budget_used += s * r
            n_rungs += 1

            # Update the rankings of the current models based on their respective metric values
            rankings[:s] = sorted(
                rankings[:s],
                key=lambda i: metrics[i].get(),
                reverse=metric.bigger_is_better
            )

            # Determine how many models to keep for the current rung
            cutoff = math.ceil(s / eta)

            _print('\t'.join((
                f'[{n_rungs}]',
                f'{s - cutoff} removed',
                f'{cutoff} left',
                f'{r} iterations',
                f'budget used: {budget_used}',
                f'budget left: {budget - budget_used}',
                f'best {metrics[rankings[0]]}',
            )))

            # Stop if there only one model left
            s = cutoff
            if s == 1:
                break

            # Determine where the next rung is located
            r = math.floor(budget / (s * math.ceil(math.log(n, eta))))

    if i < r:

        _print('Not enough data to reach the next rung, ' +
               f'{budget - budget_used} of the budget is still left')

        budget_used += i * s
        rankings[:s] = sorted(
            rankings[:s],
            key=lambda i: metrics[i].get(),
            reverse=metric.bigger_is_better
        )

    return param_grid[rankings[0]]
