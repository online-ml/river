import collections
import inspect
import typing

import numpy as np
import pandas as pd
import torch

from river import base

__all__ = ["PyTorch2RiverBase", "PyTorch2RiverRegressor", "PyTorch2RiverClassifier"]


class PyTorch2RiverBase(base.Estimator):
    """An estimator that integrates neural Networks from PyTorch."""

    def __init__(
        self,
        build_fn,
        loss_fn: typing.Type[torch.nn.modules.loss._Loss],
        optimizer_fn: typing.Type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate=1e-3,
        seed=42,
        **net_params,
    ):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = loss_fn()
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.net = None

    @classmethod
    def _unit_test_params(cls):
        def build_torch_linear_regressor(n_features):
            net = torch.nn.Sequential(
                torch.nn.Linear(n_features, 1), torch.nn.Sigmoid()
            )
            return net

        yield {
            "build_fn": build_torch_linear_regressor,
            "loss_fn": torch.nn.MSELoss,
            "optimizer_fn": torch.optim.SGD,
        }

    @classmethod
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.

        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def _learn_one(self, x: torch.Tensor, y: torch.Tensor):
        self.net.zero_grad()
        y_pred = self.net(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x: dict, y: base.typing.ClfTarget):
        """Update the model with a set of features `x` and a label `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A label.

        Returns
        -------
        self

        """
        if self.net is None:
            self._init_net(n_features=len(list(x.values())))

        x = torch.Tensor([list(x.values())])
        y = torch.Tensor([[y]])
        self._learn_one(x=x, y=y)
        return self

    def _filter_torch_params(self, fn, override=None):
        """Filters `torch_params` and returns those in `fn`'s arguments.

        Parameters
        ----------
        fn
            arbitrary function
        override
            dictionary, values to override `torch_params`

        Returns
        -------
        res
            dictionary containing variables in both and fn's arguments

        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def _init_net(self, n_features):
        self.net = self.build_fn(
            n_features=n_features, **self._filter_torch_params(self.build_fn)
        )
        # Only optimizers with learning rate as parameter are supported, needs to be fixed
        self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)


class PyTorch2RiverClassifier(PyTorch2RiverBase, base.Classifier):
    """A river classifier that integrates neural Networks from PyTorch.

    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    learning_rate
    net_params

    Examples
    --------

    >>> from river import compat
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from torch import nn
    >>> from torch import optim
    >>> from torch import manual_seed

    >>> _ = manual_seed(0)

    >>> def build_torch_mlp_classifier(n_features):
    ...     net = nn.Sequential(
    ...         nn.Linear(n_features, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 1),
    ...         nn.Sigmoid()
    ...     )
    ...     return net
    ...
    >>> model = compat.PyTorch2RiverClassifier(
    ...     build_fn= build_torch_mlp_classifier,
    ...     loss_fn=nn.BCELoss,
    ...     optimizer_fn=optim.Adam,
    ...     learning_rate=1e-3
    ... )
    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric)
    Accuracy: 74.38%

    """

    def __init__(
        self,
        build_fn,
        loss_fn: typing.Type[torch.nn.modules.loss._Loss],
        optimizer_fn: typing.Type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate=1e-3,
        **net_params,
    ):
        self.classes = collections.Counter()
        self.n_classes = 1
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            learning_rate=learning_rate,
            **net_params,
        )

    def _update_classes(self):
        self.n_classes = len(self.classes)
        layers = list(self.net.children())
        # Get last trainable layer
        i = -1
        layer_to_convert = layers[i]
        while not hasattr(layer_to_convert, "weight"):
            layer_to_convert = layers[i]
            i -= 1
        if i == -1:
            i = -2
        # Get first Layers
        new_net = list(self.net.children())[: i + 1]
        new_layer = torch.nn.Linear(
            in_features=layer_to_convert.in_features, out_features=self.n_classes
        )
        # Copy the original weights back
        with torch.no_grad():
            new_layer.weight[:-1, :] = layer_to_convert.weight
            new_layer.weight[-1:, :] = torch.mean(layer_to_convert.weight, 0)
        # Append new Layer
        new_net.append(new_layer)
        # Add non trainable layers
        if i + 1 < -1:
            for layer in layers[i + 2 :]:
                new_net.append(layer)
        self.net = torch.nn.Sequential(*new_net)
        self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        self.classes.update([y])

        # check if model is initialized
        if self.net is None:
            self._init_net(len(list(x.values())))

        # check last layer and update if needed
        if len(self.classes) != self.n_classes:
            self._update_classes()

        # training process
        proba = {c: 0.0 for c in self.classes}
        proba[y] = 1.0
        x = list(x.values())
        y = list(proba.values())

        x = torch.Tensor([x])
        y = torch.Tensor([y])
        self._learn_one(x=x, y=y)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(list(x.values())))
        x = torch.Tensor(list(x.values()))
        yp = self.net(x).detach().numpy()
        proba = {c: 0.0 for c in self.classes}
        for idx, val in enumerate(self.classes):
            proba[val] = yp[idx]
        return proba

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.net is None:
            self._init_net(len(X.columns))
        x = torch.Tensor(list(X.to_numpy()))
        yp = self.net(x).detach().numpy()
        proba = {c: [0.0] * len(X) for c in self.classes}
        for idx, val in enumerate(self.classes):
            proba[val] = yp[idx]
        return pd.DataFrame(proba)


class PyTorch2RiverRegressor(PyTorch2RiverBase, base.MiniBatchRegressor):
    """Compatibility layer from PyTorch to River for regression.

    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    learning_rate
    net_params

    Examples
    --------

    >>> from river import compat
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from torch import nn
    >>> from torch import optim

    >>> _ = torch.manual_seed(0)

    >>> dataset = datasets.TrumpApproval()

    >>> def build_torch_mlp_regressor(n_features):
    ...     net = nn.Sequential(
    ...         nn.Linear(n_features, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 1)
    ...     )
    ...     return net
    ...
    >>> model = compat.PyTorch2RiverRegressor(
    ...     build_fn= build_torch_mlp_regressor,
    ...     loss_fn=nn.MSELoss,
    ...     optimizer_fn=optim.Adam,
    ... )

    >>> metric = metrics.MAE()

    >>> metric = evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric)
    >>> round(metric.get(), 2)
    78.98

    """

    def __init__(
        self,
        build_fn,
        loss_fn: typing.Type[torch.nn.modules.loss._Loss],
        optimizer_fn: typing.Type[torch.optim.Optimizer],
        learning_rate=1e-3,
        **net_params,
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            learning_rate=learning_rate,
            **net_params,
        )

    def learn_many(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        if self.net is None:
            self._init_net(n_features=len(X.columns))

        x = torch.Tensor(X.to_numpy())
        y = torch.Tensor([y])
        self._learn_one(x=x, y=y)
        return self

    def predict_one(self, x):
        if self.net is None:
            self._init_net(len(x))
        x = torch.Tensor(list(x.values()))
        return self.net(x).item()

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        if self.net is None:
            self._init_net(len(X.columns))
        x = torch.Tensor(X.to_numpy())
        return pd.Series(self.net(x).item())
