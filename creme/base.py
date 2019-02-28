"""
Base classes used throughout the library.
"""
import abc


class Regressor:

    @abc.abstractmethod
    def fit_one(x: dict, y: float) -> float:
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters:
            x (dict)
            y (float)

        Returns:
            float: The estimated target value of ``x`` before seeing ``y``.

        """

    @abc.abstractmethod
    def predict_one(x: dict) -> float:
        """Predicts the target value of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            float

        """


class BinaryClassifier:

    @abc.abstractmethod
    def fit_one(x: dict, y: bool) -> float:
        """Fits to a set of features ``x`` and a boolean target ``y``.

        Parameters:
            x (dict)
            y (bool)

        Returns:
            float: The estimated probability of ``x`` before seeing ``y``.

        """

    @abc.abstractmethod
    def predict_proba_one(x: dict) -> float:
        """Predicts the probability output of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            float

        """

    @abc.abstractmethod
    def predict_one(x: dict) -> bool:
        """Predicts the target value of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            bool

        """


class MultiClassifier:

    @abc.abstractmethod
    def fit_one(x: dict, y: str) -> dict:
        """Fits to a set of features ``x`` and a string target ``y``.

        Parameters:
            x (dict)
            y (str)

        Returns:
            float: The estimated class probabilities of ``x`` before seeing ``y``.

        """

    @abc.abstractmethod
    def predict_proba_one(x: dict) -> dict:
        """Predicts the class probabilities of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            dict: A ``dict`` mapping classes to probabilities.

        """

    @abc.abstractmethod
    def predict_one(x: dict) -> str:
        """Predicts the class of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            str

        """


class Transformer:

    @abc.abstractmethod
    def fit_one(x: dict, y=None) -> dict:
        """Fits to a set of features ``x ` and an optinal target ``y``.

        Parameters:
            x (dict)
            y (optional)

        Returns:
            dict: The transformation applied to ``x``.

        """

    @abc.abstractmethod
    def transform_one(x: dict) -> dict:
        """Transformes a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            dict

        """


class Clusterer:

    @abc.abstractmethod
    def fit_one(x: dict, y=None) -> int:
        """Fits to a set of features ``x``.

        Parameters:
            x (dict)
            y: Not used, only present for API consistency.

        Returns:
            int: The number of the cluster ``x`` belongs to.

        """

    @abc.abstractmethod
    def predict_one(x: dict) -> int:
        """Predicts the cluster number of a set of features ``x``

        Parameters:
            x (dict)

        Returns:
            int

        """
