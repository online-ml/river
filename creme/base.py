"""
Base classes used throughout the library.
"""
import abc


class Estimator:

    def __str__(self):
        return self.__class__.__name__


class Regressor(Estimator):
    """A regressor."""

    @abc.abstractmethod
    def fit_one(self, x: dict, y: float):
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters:
            x (dict)
            y (float)

        Returns:
            self: object

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> float:
        """Predicts the target value of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            float: The prediction.

        """


class Classifier(Estimator):

    @abc.abstractmethod
    def fit_one(self, x: dict, y: bool):
        """Fits to a set of features ``x`` and a boolean target ``y``.

        Parameters:
            x (dict)
            y (bool)

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_proba_one(self, x: dict) -> dict:
        """Predicts the probability output of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            dict

        """

    def predict_one(self, x: dict) -> bool:
        """Predicts the target value of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            bool

        """
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(self.predict_proba_one(x), key=y_pred.get)
        return None


class BinaryClassifier(Classifier):
    """A binary classifier."""


class MultiClassifier(BinaryClassifier):
    """A multi-class classifier."""


class Transformer(Estimator):
    """A transformer."""

    def fit_one(self, x: dict, y=None) -> dict:
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
        """Transforms a set of features ``x``

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
    def fit_one(self, x: dict, y=None) -> int:
        """Fits to a set of features ``x``.

        Parameters:
            x (dict)
            y: Not used, only present for API consistency.

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> int:
        """Predicts the cluster number of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            int

        """
