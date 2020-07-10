from skmultiflow.core import BaseSKMObject
from sklearn.utils import tosequence


class Pipeline(BaseSKMObject):
    """ [Experimental] Holds a set of sequential operation (transforms),
    followed by a single estimator.

    It allows for easy manipulation of datasets that may
    require several transformation processes before being used by a learner.
    Also allows for the cross-validation of several steps.

    Each of the intermediate steps should be an extension of the BaseTransform
    class, or at least implement the transform and partial_fit functions or the
    partial_fit_transform.

    The last step should be an estimator (learner), so it should implement
    partial_fit, and predict at least.

    Since it has an estimator as the last step, the Pipeline will act like
    an estimator itself, in a way that it can be directly passed to evaluation
    objects, as if it was a learner.

    Parameters
    ----------
    steps: list of tuple
        Tuple list containing the set of transforms and the final estimator.
        It doesn't need to contain a transform type object, but the estimator
        is required. Each tuple should be of the format ('name', estimator).

    Raises
    ------
    TypeError: If the intermediate steps or the final estimator do not implement
    the necessary functions for the pipeline to work, a TypeError is raised.

    NotImplementedError: Some of the functions are yet to be implemented.

    Notes
    -----
    This code is an experimental feature. Use with caution.


    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.lazy import KNNADWINClassifier
    >>> from skmultiflow.core import Pipeline
    >>> from skmultiflow.data import FileStream
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>> from skmultiflow.transform import OneHotToCategorical
    >>> # Setting up the stream
    >>> stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/"
    ...                     "streaming-datasets/master/covtype.csv")
    >>> transform = OneHotToCategorical([[10, 11, 12, 13],
    ... [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    ... 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    >>> # Setting up the classifier
    >>> classifier = KNNADWINClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40)
    >>> # Setup the pipeline
    >>> pipe = Pipeline([('transform', transform), ('passive_aggressive', classifier)])
    >>> # Setup the evaluator
    >>> evaluator = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_samples=500000)
    >>> # Evaluate
    >>> evaluator.evaluate(stream=stream, model=pipe)

    """
    _estimator_type = 'pipeline'

    def __init__(self, steps):
        # Default values
        super().__init__()
        self.steps = tosequence(steps)
        self.active = False

        self.__configure()

    def __configure(self):
        """ __configure

        Initial Pipeline configuration. Validates the Pipeline's steps.

        """
        self._validate_steps()

    def predict(self, X):
        """ predict

        Sequentially applies all transforms and then predict with last step.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list
            The predicted class label for all the samples in X.

        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt)

    def fit(self, X, y):
        """ fit

        Sequentially fit and transform data in all but last step, then fit
        the model in last step.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The data upon which the transforms/estimator will create their
            model.

        y: An array_like object of length n_samples
            Contains the true class labels for all the samples in X.

        Returns
        -------
        Pipeline
            self

        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y)
            else:
                Xt = transform.fit(Xt, y).transform(Xt)

        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """ partial_fit

        Sequentially partial fit and transform data in all but last step,
        then partial fit data in last step.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        Returns
        -------
        Pipeline
            self

        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, 'fit_transform'):
                Xt = transform.partial_fit_transform(Xt, y, classes=classes)
            else:
                Xt = transform.partial_fit(Xt, y, classes=classes).transform(Xt)

        if self._final_estimator is not None:
            if "classes" in self._final_estimator.partial_fit.__code__.co_varnames:
                self._final_estimator.partial_fit(X=Xt, y=y, classes=classes)
            else:
                self._final_estimator.partial_fit(X=Xt, y=y)
        return self

    def partial_fit_predict(self, X, y):
        """ partial_fit_predict

        Partial fits and transforms data in all but last step, then partial
        fits and predicts in the last step

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        y: An array_like object of length n_samples
            Contains the true class labels for all the samples in X

        Returns
        -------
        list
            The predicted class label for all the samples in X.

        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, "partial_fit_transform"):
                Xt = transform.partial_fit_transform(Xt, y)
            else:
                Xt = transform.partial_fit(Xt, y).transform(Xt)

        if hasattr(self._final_estimator, "partial_fit_predict"):
            return self._final_estimator.partial_fit_predict(Xt, y)
        else:
            return self._final_estimator.partial_fit(Xt, y).predict(Xt)

    def partial_fit_transform(self, X, y=None):
        """ partial_fit_transform

        Partial fits and transforms data in all but last step, then
        partial_fit in last step

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The data upon which the transforms/estimator will create their
            model.

        y: An array_like object of length n_samples
            Contains the true class labels for all the samples in X

        Returns
        -------
        Pipeline
            self

        """
        raise NotImplementedError

    def _validate_steps(self):
        """ validate_steps

        Validates all steps, guaranteeing that there's an estimator in its last step.

        Alters the value of self.active according to the validity of the steps.

        Raises
        ------
        TypeError: If the intermediate steps or the final estimator do not implement
        the necessary functions for the pipeline to work, a TypeError is raised.

        """

        names, estimators = zip(*self.steps)
        classifier = estimators[-1]
        transforms = estimators[:-1]

        self.active = True

        for t in transforms:
            if t is None:
                continue
            else:
                if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) \
                        or not hasattr(t, "transform"):
                    self.active = False
                    raise TypeError("All intermediate steps, including an evaluator, "
                                    "should implement fit and transform.")

        if classifier is not None and not hasattr(classifier, "partial_fit"):
            self.active = False

    def named_steps(self):
        """ named_steps

        Generates a dictionary to access all the steps' properties.

        Returns
        -------
        dictionary
            A steps dictionary, so that each step can be accessed by name.

        """
        return dict(self.steps)

    def get_info(self):
        info = "Pipeline:\n["
        names, estimators = zip(*self.steps)
        learner = estimators[-1]
        transforms = estimators[:-1]
        i = 0
        for t in transforms:
            try:
                if t.get_info() is not None:
                    info += t.get_info()
                    info += "\n"
                else:
                    info += 'Transform: no info available'
            except NotImplementedError:
                info += 'Transform: no info available'
            i += 1

        if learner is not None:
            try:
                if hasattr(learner, 'get_info'):
                    info += learner.get_info()
                else:
                    info += 'Learner: no info available'
            except NotImplementedError:
                info += 'Learner: no info available'
        info += "]"
        return info

    @property
    def _final_estimator(self):
        """ _final_estimator

        Easy to access estimator.

        Returns
        -------
        Extension of BaseClassifier
            The Pipeline's classifier

        """
        return self.steps[-1][-1]
