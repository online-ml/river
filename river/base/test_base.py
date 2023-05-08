from __future__ import annotations

from river import compose, datasets, linear_model, optim, preprocessing, stats, time_series


def test_clone_estimator():
    obj = linear_model.LinearRegression(l2=42)
    obj.learn_one({"x": 3}, 6)

    new = obj.clone({"l2": 21})
    assert new.l2 == 21
    assert obj.l2 == 42
    assert new.weights == {}
    assert new.weights != obj.weights


def test_clone_include_attributes():
    var = stats.Var()
    var.update(1)
    var.update(2)
    var.update(3)

    assert var._S == 2
    assert var.clone()._S == 0
    assert var.clone(include_attributes=True)._S == 2


def test_clone_pipeline():
    obj = preprocessing.StandardScaler() | linear_model.LinearRegression(l2=42)
    obj.learn_one({"x": 3}, 6)

    params = {"LinearRegression": {"l2": 21}}
    new = obj.clone(params)
    assert new["LinearRegression"].l2 == 21
    assert obj["LinearRegression"].l2 == 42
    assert new["LinearRegression"].weights == {}
    assert new["LinearRegression"].weights != obj["LinearRegression"].weights


def test_clone_idempotent():
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.Adam(), l2=0.1
    )

    trace = []
    for x, y in datasets.Phishing():
        trace.append(model.predict_proba_one(x))
        model.learn_one(x, y)

    clone = model.clone()
    for i, (x, y) in enumerate(datasets.Phishing()):
        assert clone.predict_proba_one(x) == trace[i]
        clone.learn_one(x, y)


def test_memory_usage():
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression()

    # We can't test the exact value because it depends on the platform and the Python version
    # TODO: we could create a table of expected values for each platform and Python version
    assert isinstance(model._memory_usage, str)


def test_mutate():
    """

    >>> from river import datasets, linear_model, optim, preprocessing

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LinearRegression(
    ...         optimizer=optim.SGD(3e-2)
    ...     )
    ... )

    >>> for x, y in datasets.TrumpApproval():
    ...     _ = model.predict_one(x)
    ...     model = model.learn_one(x, y)

    >>> len(model[-1].weights)
    6

    >>> model.mutate({'LinearRegression': {'optimizer': {'lr': {'learning_rate': 42}}}})
    >>> model[-1].optimizer
    SGD({'lr': Constant({'learning_rate': 42}), 'n_iterations': 1001})

    >>> model.mutate({'LinearRegression': {'optimizer': {'lr': optim.schedulers.Constant(43)}}})
    >>> model[-1].optimizer
    SGD({'lr': Constant({'learning_rate': 43}), 'n_iterations': 1001})

    >>> model.mutate({'LinearRegression': {'optimizer': optim.SGD(44)}})
    >>> model[-1].optimizer
    SGD({'lr': Constant({'learning_rate': 44}), 'n_iterations': 0})

    >>> model.mutate({'LinearRegression': {'l2': 0.123}})
    >>> model[-1].l2
    0.123

    >>> len(model[-1].weights)
    6

    >>> try:
    ...     model.mutate({'LinearRegression': {'weights': 'this is weird'}})
    ... except ValueError as e:
    ...     print(e)
    'weights' is not a mutable attribute of LinearRegression

    >>> try:
    ...     model.mutate({'LinearRegression': {'l3': 123}})
    ... except ValueError as e:
    ...     print(e)
    'l3' is not an attribute of LinearRegression

    """


def test_clone_positional_args():
    assert compose.Select(1, 2, 3).clone().keys == {1, 2, 3}
    assert compose.Discard("a", "b", "c").clone().keys == {"a", "b", "c"}
    assert compose.SelectType(float, int).clone().types == (float, int)


def test_clone_nested_pipeline():
    model = time_series.SNARIMAX(
        p=2,
        d=1,
        q=3,
        regressor=(
            preprocessing.StandardScaler()
            | linear_model.LinearRegression(optimizer=optim.SGD(3e-2))
        ),
    )
    assert model.clone()._get_params() == model._get_params()
