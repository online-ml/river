from __future__ import annotations

import abc
import typing
from typing import Any

import narwhals.stable.v2 as nw

from river import base

from . import estimator

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries


class Classifier(estimator.Estimator):
    """A classifier."""

    @abc.abstractmethod
    def learn_one(self, x: dict[base.typing.FeatureName, Any], y: base.typing.ClfTarget) -> None:
        """Update the model with a set of features `x` and a label `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A label.

        """

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> dict[base.typing.ClfTarget, float]:
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        A dictionary that associates a probability which each label.

        """

        # Some classifiers don't have the ability to output probabilities, and instead only
        # predict labels directly. Therefore, we cannot impose predict_proba_one as an abstract
        # method that each classifier has to implement. Instead, we raise an exception to indicate
        # that a classifier does not support predict_proba_one.
        raise NotImplementedError

    def predict_one(
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> base.typing.ClfTarget | None:
        """Predict the label of a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The predicted label.

        """

        # The following code acts as a default for each classifier, and may be overridden on an
        # individual basis.
        y_pred = self.predict_proba_one(x, **kwargs)
        if y_pred:
            return max(y_pred, key=y_pred.get)  # type: ignore
        return None

    @property
    def _multiclass(self) -> bool:
        return False

    @property
    def _supervised(self) -> bool:
        return True


class MiniBatchClassifier(Classifier):
    """A classifier that can operate on mini-batches."""

    @abc.abstractmethod
    def learn_many(self, X: IntoDataFrame, y: IntoSeries) -> None:
        """Update the model with a mini-batch of features `X` and boolean targets `y`.

        Parameters
        ----------
        X
            A dataframe of features. Any narwhals-supported eager backend is accepted
            (pandas, polars, PyArrow, etc.).
        y
            A series of boolean target values.

        """

    def predict_proba_many(self, X: IntoDataFrame) -> IntoDataFrame:
        """Predict the outcome probabilities for each given sample.

        Parameters
        ----------
        X
            A dataframe of features. Any narwhals-supported eager backend is accepted
            (pandas, polars, PyArrow, etc.).

        Returns
        -------
        A dataframe with probabilities of `True` and `False` for each sample, in the same
        backend as `X`.

        """

        # Some classifiers don't have the ability to output probabilities, and instead only
        # predict labels directly. Therefore, we cannot impose predict_proba_many as an abstract
        # method that each classifier has to implement. Instead, we raise an exception to indicate
        # that a classifier does not support predict_proba_many.
        raise NotImplementedError

    def predict_many(self, X: IntoDataFrame) -> IntoSeries:
        """Predict the outcome for each given sample.

        Parameters
        ----------
        X
            A dataframe of features. Any narwhals-supported eager backend is accepted
            (pandas, polars, PyArrow, etc.).

        Returns
        -------
        The predicted labels, in the same backend as `X`.

        """

        # The following code acts as a default for each classifier, and may be overridden on an
        # individual basis.
        import numpy as np

        proba_native = self.predict_proba_many(X)
        proba_nw = nw.from_native(proba_native, eager_only=True)
        # Equivalent to pandas .empty: no rows or no columns (e.g. untrained model).
        # Return the probability frame as-is to preserve the caller's backend and index.
        if len(proba_nw) == 0 or len(proba_nw.columns) == 0:
            return proba_native  # type: ignore[return-value]
        arr = proba_nw.to_numpy()
        # Read native column labels so non-string class labels (e.g. int) are preserved.
        native_proba = nw.to_native(proba_nw)
        native_cols = list(
            native_proba.columns
            if hasattr(native_proba, "columns")
            else proba_nw.columns
        )
        labels = np.asarray(native_cols)[arr.argmax(axis=1)]
        Xnw = nw.from_native(X, eager_only=True)
        ns = nw.get_native_namespace(Xnw)
        series = nw.new_series(name=None, values=labels, backend=ns)  # type: ignore[arg-type]
        return typing.cast("IntoSeries", series.to_native())
