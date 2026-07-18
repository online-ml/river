from __future__ import annotations

import typing

import narwhals.stable.v2 as nw
import numpy as np

from river import base, linear_model, utils

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries

__all__ = ["OneVsRestClassifier"]


class OneVsRestClassifier(base.Wrapper, base.Classifier):
    """One-vs-the-rest (OvR) multiclass strategy.

    This strategy consists in fitting one binary classifier per class. Because we are in a
    streaming context, the number of classes isn't known from the start. Hence, new classifiers are
    instantiated on the fly. Likewise, the predicted probabilities will only include the classes
    seen up to a given point in time.

    Note that this classifier supports mini-batches as well as single instances.

    The computational complexity for both learning and predicting grows linearly with the number of
    classes. If you have a very large number of classes, then you might want to consider using an
    `multiclass.OutputCodeClassifier` instead.

    Parameters
    ----------
    classifier
        A binary classifier, although a multi-class classifier will work too.

    Attributes
    ----------
    classifiers : dict
        A mapping between classes and classifiers.

    Examples
    --------

    >>> import pandas as pd
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multiclass
    >>> from river import preprocessing

    >>> dataset = datasets.ImageSegments()

    >>> scaler = preprocessing.StandardScaler()
    >>> ovr = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    >>> model = scaler | ovr

    >>> metric = metrics.MacroF1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MacroF1: 77.46%

    This estimator also supports mini-batching. The whole pipeline — the `StandardScaler` and the
    one-vs-rest `LogisticRegression`s — is trained and queried on mini-batches of a dataframe:

    >>> model = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(
    ...     linear_model.LogisticRegression()
    ... )

    >>> metric = metrics.MacroF1()

    >>> for X in pd.read_csv(dataset.path, chunksize=64):
    ...     y = X.pop('category')
    ...     y_pred = model.predict_many(X)
    ...     model.learn_many(X, y)
    ...     for y_true, y_p in zip(y, y_pred):
    ...         if y_p is not None:
    ...             metric.update(y_true, y_p)

    >>> metric
    MacroF1: 59.77%

    The mini-batch methods are dataframe-agnostic: any narwhals-supported eager backend
    (pandas, polars, pyarrow, ...) can be passed to `learn_many`/`predict_many`, and the
    predictions are returned in that same backend.

    """

    def __init__(self, classifier: base.Classifier):
        self.classifier = classifier
        self.classifiers: dict[base.typing.ClfTarget, base.Classifier] = {}
        self._y_name: str | None = None

    @property
    def _wrapped_model(self):
        return self.classifier

    @property
    def _multiclass(self):
        return True

    @classmethod
    def _unit_test_params(cls):
        yield {"classifier": linear_model.LogisticRegression()}

    def learn_one(self, x, y, **kwargs):
        # Instantiate a new binary classifier if the class is new
        if y not in self.classifiers:
            self.classifiers[y] = self.classifier.clone()

        # Train each label's associated classifier
        for label, model in self.classifiers.items():
            model.learn_one(x, y == label, **kwargs)

    def predict_proba_one(self, x, **kwargs):
        y_pred = {}
        total = 0.0

        for label, model in self.classifiers.items():
            yp = model.predict_proba_one(x, **kwargs)[True]
            y_pred[label] = yp
            total += yp

        if total:
            return {label: votes / total for label, votes in y_pred.items()}
        return {label: 1 / len(y_pred) for label in y_pred}

    def learn_many(self, X: IntoDataFrame, y: IntoSeries, **kwargs) -> None:
        # narwhals at the boundary: the per-label target is a boolean series built with the input
        # series' own backend, so each binary sub-classifier sees a native series it understands.
        ynw = utils.dataframe.into_series(y)
        self._y_name = ynw.name

        # Instantiate a new binary classifier for the classes that have not yet been seen. The
        # appearance order is preserved so the class set matches the single-instance path.
        for label in ynw.unique(maintain_order=True).to_list():
            if label not in self.classifiers:
                self.classifiers[label] = self.classifier.clone()

        # Train each label's associated classifier against the one-vs-rest boolean target.
        for label, model in self.classifiers.items():
            typing.cast("base.MiniBatchClassifier", model).learn_many(
                X, (ynw == label).to_native(), **kwargs
            )

    def _positive_probas(
        self, X, **kwargs
    ) -> tuple[nw.DataFrame[IntoDataFrame], list[base.typing.ClfTarget], np.ndarray]:
        """Return ``(Xnw, labels, P)`` with the raw positive-class probabilities.

        Column ``j`` of ``P`` is the (un-normalised) probability that ``labels[j]`` is the
        positive class. Each binary sub-classifier exposes that probability under the ``True``
        column, which narwhals reports as the boolean ``True`` on pandas and the string
        ``"True"`` elsewhere; either way ``str(column) == "True"`` locates it (selecting by a
        boolean key is rejected by narwhals, so the column is taken positionally).
        """
        Xnw = utils.dataframe.into_frame(X)
        labels = list(self.classifiers)
        columns = []
        for clf in self.classifiers.values():
            proba = utils.dataframe.into_frame(
                typing.cast("base.MiniBatchClassifier", clf).predict_proba_many(X, **kwargs)
            )
            positive = next(j for j, col in enumerate(proba.columns) if str(col) == "True")
            columns.append(utils.dataframe.to_numpy(proba)[:, positive])

        P = np.column_stack(columns) if columns else np.empty((len(Xnw), 0))
        return Xnw, labels, P

    def predict_proba_many(self, X: IntoDataFrame, **kwargs) -> IntoDataFrame:
        Xnw, labels, P = self._positive_probas(X, **kwargs)
        totals = P.sum(axis=1, keepdims=True)
        # Normalise each row to sum to 1. `where` leaves all-zero rows as NaN — matching the old
        # pandas `df.div(df.sum(axis="columns"))` — without triggering a divide-by-zero warning.
        P = np.divide(P, totals, out=np.full_like(P, np.nan), where=totals != 0)
        return utils.dataframe.to_native_frame(
            {label: P[:, j] for j, label in enumerate(labels)}, like=Xnw
        )

    def predict_many(self, X: IntoDataFrame, **kwargs) -> IntoSeries:
        Xnw, labels, P = self._positive_probas(X, **kwargs)
        if not labels:
            return utils.dataframe.to_native_series([None] * len(Xnw), name=self._y_name, like=Xnw)
        # Row-normalisation scales every class by the same per-row constant, so the winner is the
        # same as over the raw probabilities. `argmax` breaks ties on the first column, matching
        # the old pandas `idxmax(axis="columns")`.
        winners = [labels[int(i)] for i in np.argmax(P, axis=1)]
        return utils.dataframe.to_native_series(winners, name=self._y_name, like=Xnw)
