"""Base interfaces.

Every estimator in ``creme`` is a class, and as such inherits from at least one base interface.
These are used to categorize, organize, and standardize the many estimators that ``creme``
contains.

"""
import abc
import collections
import inspect
import typing

from . import utils


__all__ = [
    'BinaryClassifier',
    'Clusterer',
    'Ensemble',
    'Estimator',
    'Wrapper',
    'MultiClassifier',
    'MultiOutputClassifier',
    'MultiOutputRegressor',
    'Regressor',
    'Transformer',
    'OutlierDetector'
]


# Input and output types
Label = typing.Union[bool, str, int]
Proba = float
Probas = typing.Dict[Label, Proba]


DEFAULT_TAGS = {
    'handles_text': False,
    'requires_positive_data': False,
    'handles_categorical_features': False,
    'poor_score': False
}


def _update_if_consistent(dict1, dict2):
    common_keys = set(dict1.keys()).intersection(dict2.keys())
    for key in common_keys:
        if dict1[key] != dict2[key]:
            raise TypeError(f'Inconsistent values for tag {key}: {dict1[key]} != {dict2[key]}')
    dict1.update(dict2)
    return dict1


class Estimator:
    """An estimator."""

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return utils.pretty_format_class(self)

    def _more_tags(self) -> dict:
        """Specific tags for this estimator."""
        return {}

    def _get_tags(self) -> dict:
        """Returns the estimator's tags."""

        tags = {}

        for base_class in inspect.getmro(self.__class__):
            if hasattr(base_class, '_more_tags') and base_class != self.__class__:
                tags = _update_if_consistent(tags, base_class._more_tags(self))

        if hasattr(self, '_more_tags'):
            tags = _update_if_consistent(tags, self._more_tags())

        return {**DEFAULT_TAGS, **tags}


class Regressor(Estimator):
    """A regressor."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: float) -> 'Regressor':
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters:
            x (dict)
            y (float)

        Returns:
            self: object

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> float:
        """Predicts the target value of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            float: The prediction.

        """


class Classifier(Estimator):
    """A classifier."""

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> Probas:
        """Predicts the probability output of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            dict of floats

        """

    def predict_one(self, x: dict) -> Label:
        """Predicts the target value of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            Label

        """
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(self.predict_proba_one(x), key=y_pred.get)
        return None


class BinaryClassifier(Classifier):
    """A binary classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: bool) -> 'BinaryClassifier':
        """Fits to a set of features ``x`` and a boolean target ``y``.

        Parameters:
            x (dict)
            y (bool)

        Returns:
            self

        """


class MultiClassifier(BinaryClassifier):
    """A multi-class classifier."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: Label) -> 'MultiClassifier':
        """Fits to a set of features ``x`` and a label ``y``.

        Parameters:
            x (dict)
            y (Label)

        Returns:
            self

        """


class Transformer(Estimator):
    """A transformer."""

    def fit_one(self, x: dict, y=None) -> 'Transformer':
        """Fits to a set of features ``x`` and an optional target ``y``.

        A lot of transformers don't actually have to do anything during the ``fit_one`` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the ``fit_one`` can override this
        method.

        Parameters:
            x (dict)
            y (optional)

        Returns:
            self

        """
        return self

    @abc.abstractmethod
    def transform_one(self, x: dict) -> dict:
        """Transforms a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            dict

        """

    @property
    def is_supervised(self) -> bool:
        """Indicates if the transformer uses the target ``y`` or not.

        Supervised transformers have to be handled differently from unsupervised transformers in an
        online setting. This is especially true for target encoding where leakage can easily occur.
        Most transformers are unsupervised and so this property returns by default ``False``.
        Transformers that are supervised must override this property in their definition.

        """
        return False

    def __or__(self, other):
        """Merges with another Transformer into a Pipeline."""
        from . import compose
        if isinstance(other, compose.Pipeline):
            return other.__ror__(self)
        return compose.Pipeline([self, other])

    def __ror__(self, other):
        """Merges with another Transformer into a Pipeline."""
        from . import compose
        if isinstance(other, compose.Pipeline):
            return other.__or__(self)
        return compose.Pipeline([other, self])

    def __add__(self, other):
        """Merges with another Transformer into a TransformerUnion."""
        from . import compose
        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion([self, other])

    def __radd__(self, other):
        """Merges with another Transformer into a TransformerUnion."""
        from . import compose
        if isinstance(other, compose.TransformerUnion):
            return other.__add__(self)
        return compose.TransformerUnion([other, self])


class Clusterer(Estimator):
    """A clusterer."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y=None) -> 'Clusterer':
        """Fits to a set of features ``x``.

        Parameters:
            x (dict)
            y: Not used, only present for API consistency.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> int:
        """Predicts the cluster number of a set of features ``x``.

        Parameters:
            x (dict)

        Returns:
            int

        """


class MultiOutputEstimator(Estimator):
    """A multi-output estimator."""


class MultiOutputClassifier(MultiOutputEstimator):
    """A multi-output classifier."""

    def fit_one(self, x: dict, y: typing.Dict[str, Label]) -> 'MultiOutputClassifier':
        """Fits to a set of features ``x`` and a set of labels ``y``.

        Parameters:
            x (dict)
            y (dict of Label)

        Returns:
            self

        """

    def predict_one(self, x: dict) -> typing.Dict[str, Label]:
        """Given a set of features ``x``, predicts a label for each output.

        Parameters:
            x (dict)

        Returns:

        """
        y_pred = self.predict_proba_one(x)
        return {
            c: max(y_pred[c], key=y_pred[c].get)
            for c in y_pred
        }

    def predict_proba_one(self, x: dict) -> typing.Dict[str, Probas]:
        """Given a set of features ``x``, predicts a the probability of each label for each output.

        Parameters:
            x (dict)

        Returns:

        """


class MultiOutputRegressor(MultiOutputEstimator):
    """A multi-output regressor."""

    def fit_one(self, x: dict, y: typing.Dict[str, float]) -> 'MultiOutputRegressor':
        """Fits to a set of features ``x`` and a set of outputs ``y``.

        Parameters:
            x (dict)
            y (dict of Label)

        Returns:
            self

        """

    def predict_one(self, x: dict) -> typing.Dict[str, float]:
        """Given a set of features ``x``, predicts a label for each output.

        Parameters:
            x (dict)

        Returns:
            dict

        """


class Wrapper(abc.ABC):
    """A wrapper model."""

    @abc.abstractproperty
    def _model(self):
        """Provides access to the wrapped model."""

    def __str__(self):
        return f'{type(self).__name__}({self._model})'


class Ensemble(Estimator, collections.UserList):
    """An ensemble model."""


class OutlierDetector(Estimator):

    @abc.abstractmethod
    def fit_one(self, x: dict) -> 'OutlierDetector':
        """Updates the model."""

    @abc.abstractmethod
    def score_one(self, x: dict) -> float:
        """Returns an outlier score.

        The range of the score depends on each model. Some models will output anomaly scores
        between 0 and 1, others will not. In any case, the lower the score, the more likely it is
        that ``x`` is an anomaly.

        """
