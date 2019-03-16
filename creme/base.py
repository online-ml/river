"""
Base classes used throughout the library.
"""
import abc


class Regressor:

    @abc.abstractmethod
    def fit_one(self, x: dict, y: float):
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters:
            x (dict)
            y (float)

        Returns:
            self

        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> float:
        """Predicts the target value of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            float

        """

    def fit_predict_one(self, x: dict, y: float) -> float:
        y_pred = self.predict_one(x)
        self.fit_one(x, y)
        return y_pred


class BinaryClassifier:

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

    def fit_predict_proba_one(self, x: dict, y: float) -> dict:
        y_pred = self.predict_proba_one(x)
        self.fit_one(x, y)
        return y_pred

    def fit_predict_one(self, x: dict, y: float) -> bool:
        y_pred = self.predict_one(x)
        self.fit_one(x, y)
        return y_pred


class MultiClassifier(BinaryClassifier):
    """A MultiClassifier can handle more than two classes."""


class Transformer:

    def fit_one(self, x: dict, y=None) -> dict:
        """Fits to a set of features ``x ` and an optinal target ``y``.

        A lot of transformers don't actually have to do anything during the ``fit_one`` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that do however do something during the ``fit_one`` can override this
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
        """Transformes a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            dict

        """

    @property
    def is_supervised(self) -> bool:
        """Indicates if the transformer used the target ``y`` or not.

        Supervised transformers have to be handled differently from unsupervised transformers in an
        online setting. This is especially true for target encoding where leakage can easily occur.
        Most transformers are unsupervised and so this property returns by default ``False``.
        Transformers that are supervised must override this property in their definition.

        """
        return False

    def fit_transform_one(self, x: dict, y=None):
        if self.is_supervised:
            y_pred = self.transform_one(x)
            self.fit_one(x, y)
            return y_pred
        return self.fit_one(x, y).transform_one(x)

    def __add__(self, other):
        from . import compose
        return compose.Pipeline([self, other])

    def __or__(self, other):
        from . import compose
        return compose.FeatureUnion([self, other])


class Clusterer:

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
