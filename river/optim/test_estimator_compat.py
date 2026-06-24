"""Every optimizer should work with every estimator that accepts one.

These estimators expose the optimizer as a hyperparameter, so any concrete optimizer ought
to be able to drive any of them. This used to silently break for estimators whose weights are
stored as NumPy arrays (e.g. the factorization machines): some optimizers assumed dict-like
weights and raised at ``learn_one`` time. This test runs the full ``optimizer`` x ``estimator``
matrix on a small dummy stream to guard against that.

A handful of combinations are genuinely unsupported (see ``_xfail_reason``) and are marked as
strict xfails, so the matrix stays exhaustive and we get told if one ever starts working.
"""

from __future__ import annotations

import importlib
import inspect
import random

import pytest

from river import base, optim
from river.anomaly.base import AnomalyDetector
from river.reco.base import Ranker

# __init__ parameters through which an optimizer can be injected.
OPTIMIZER_PARAMS = frozenset(
    {"optimizer", "weight_optimizer", "latent_optimizer", "int_weight_optimizer"}
)

# Optimizers whose updates rely on scalar-only operations (element-wise max/clamp/threshold, or a
# Hessian). They cannot drive models that keep a *vector* of weights per key, such as the
# recommenders.
SCALAR_ONLY_OPTIMIZERS = frozenset({"AdaBound", "AdaMax", "AMSGrad", "FTRLProximal", "Newton"})


def iter_optimizers():
    """All concrete optimizers, instantiated with their default hyperparameters."""
    for name in sorted(dir(optim)):
        obj = getattr(optim, name)
        if not (inspect.isclass(obj) and issubclass(obj, optim.base.Optimizer)):
            continue
        if obj is optim.base.Optimizer:
            continue
        try:
            yield obj()
        except TypeError:
            # Meta-optimizers (e.g. Averager) wrap another optimizer.
            yield obj(optim.SGD())


def iter_optimizer_estimators():
    """All concrete estimators that accept an optimizer, with their optimizer params."""
    seen: set[type] = set()
    for submodule in importlib.import_module("river.api").__all__:
        module = importlib.import_module(f"river.{submodule}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, base.Estimator) or inspect.isabstract(cls):
                continue
            if cls in seen:
                continue
            # MLPRegressor is deprecated and requires an explicit architecture; skip it.
            if cls.__name__ == "MLPRegressor":
                continue
            try:
                params = set(inspect.signature(cls.__init__).parameters)
            except (TypeError, ValueError):
                continue
            optimizer_params = OPTIMIZER_PARAMS & params
            if optimizer_params:
                seen.add(cls)
                yield cls, optimizer_params


def _xfail_reason(optimizer, estimator_cls) -> str | None:
    """Reason a given (optimizer, estimator) pair is expected to fail, or None if it should work."""
    name = optimizer.__class__.__name__
    is_factorization_machine = estimator_cls.__module__.startswith("river.facto")
    # Models with latent factors keep a vector of weights per key (the recommenders apart from the
    # bias-only `Baseline`, which works with every optimizer).
    has_latent_vectors = "n_factors" in inspect.signature(estimator_cls.__init__).parameters
    keeps_vector_weights_per_key = issubclass(estimator_cls, Ranker) and has_latent_vectors

    if name == "Averager" and is_factorization_machine:
        return (
            "Averager keeps a single running average of the weights and returns it, so it cannot "
            "drive the many independent latent vectors a factorization machine updates through "
            "one shared optimizer instance."
        )
    if keeps_vector_weights_per_key and (name in SCALAR_ONLY_OPTIMIZERS or name == "Averager"):
        return (
            "Recommenders update a vector of weights per key; this optimizer only supports "
            "scalar-valued weights."
        )
    return None


def _exercise(model):
    """Run ``model`` on a small dummy stream appropriate to its kind."""
    rng = random.Random(42)

    if isinstance(model, Ranker):
        users, items = ["Alice", "Bob", "Carol"], ["x", "y", "z"]
        for _ in range(20):
            user, item = rng.choice(users), rng.choice(items)
            model.predict_one(user, item)
            model.learn_one(user, item, rng.uniform(0, 5))
        return

    if isinstance(model, AnomalyDetector):
        for _ in range(20):
            x = {f"f{j}": rng.uniform(-1, 1) for j in range(5)}
            model.score_one(x)
            model.learn_one(x)
        return

    is_classifier = isinstance(model, base.Classifier)
    for _ in range(20):
        x = {f"f{j}": rng.uniform(-1, 1) for j in range(5)}
        y = rng.choice([False, True]) if is_classifier else rng.uniform(-1, 1)
        model.predict_one(x)
        model.learn_one(x, y)


def _matrix():
    for estimator_cls, optimizer_params in iter_optimizer_estimators():
        for optimizer in iter_optimizers():
            reason = _xfail_reason(optimizer, estimator_cls)
            marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
            yield pytest.param(
                estimator_cls,
                optimizer_params,
                optimizer,
                id=f"{optimizer.__class__.__name__}-{estimator_cls.__name__}",
                marks=marks,
            )


@pytest.mark.parametrize("estimator_cls, optimizer_params, optimizer", list(_matrix()))
def test_optimizer_works_with_estimator(estimator_cls, optimizer_params, optimizer):
    kwargs = {param: optimizer.clone() for param in optimizer_params}
    model = estimator_cls(**kwargs)
    _exercise(model)
