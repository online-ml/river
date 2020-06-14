import numpy as np
import copy

from sklearn.preprocessing import normalize

from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import check_random_state
from skmultiflow.utils import get_dimensions


class HalfSpaceTrees(BaseSKMObject, ClassifierMixin):
    """Half--Space Trees.

    Implementation of the Streaming Half--Space--Trees (HS--Trees) [1]_,
    a fast one-class anomaly detector for evolving data streams. It requires
    only normal data for training and works well when anomalous data are rare.
    The model features an ensemble of random HS--Trees, and the tree structure
    is constructed without any data. This makes the method highly efficient
    because it requires no model restructuring when adapting to evolving data
    streams.

    Parameters
    ----------
    n_estimators: int, optional (default=25)
       Number of trees in the ensemble.
       't' in the original paper.

    window_size: int, optional (default=250)
        The window size of the stream.
        'Psi' in the original paper.

    depth: int, optional (default=15)
        The maximum depth of the trees in the ensemble.
        'maxDepth' in the original paper.

    size_limit: int, optional (default=50)
        The minimum mass required in a node (as a fraction of the window size)\
        to calculate the anomaly score.
        'sizeLimit' in the original paper.
        A good setting is ``0.1 * window_size``

    anomaly_threshold: double, optional (default=0.5)
        The threshold for declaring anomalies.
        Any instance prediction probability above this threshold will be
        declared as an anomaly.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    .. [1] S.C.Tan, K.M.Ting, and T.F.Liu, “Fast anomaly detection for
       streaming data,” in IJCAI Proceedings - International Joint Conference
       on Artificial Intelligence, 2011, vol. 22, no. 1, pp. 1511–1516.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import AnomalySineGenerator
    >>> from skmultiflow.anomaly_detection import HalfSpaceTrees
    >>> # Setup a data stream
    >>> stream = AnomalySineGenerator(random_state=1, n_samples=1000, n_anomalies=250)
    >>> # Setup Half-Space Trees estimator
    >>> half_space_trees = HalfSpaceTrees(random_state=1)
    >>> # Setup variables to control loop and track performance
    >>> max_samples = 1000
    >>> n_samples = 0
    >>> true_positives = 0
    >>> detected_anomalies = 0
    >>> # Train the estimator(s) with the samples provided by the data stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = half_space_trees.predict(X)
    >>>     if y[0] == 1:
    >>>         true_positives += 1
    >>>         if y_pred[0] == 1:
    >>>             detected_anomalies += 1
    >>>     half_space_trees.partial_fit(X, y)
    >>>     n_samples += 1
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Half-Space Trees correctly detected {} out of {} anomalies'.
    >>>       format(detected_anomalies, true_positives))
    """

    def __init__(self,
                 window_size=250,
                 depth=15,
                 n_estimators=25,
                 size_limit=50,
                 anomaly_threshold=0.5,
                 random_state=None):

        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.n_estimators = n_estimators
        self.min_values = []
        self.max_values = []
        self.ensemble = []
        self.n_features = -1
        self.size_limit = size_limit
        self.samples_seen = 0
        self.anomaly_threshold = anomaly_threshold
        self.is_learning_phase_on = True
        self.random_state = random_state
        self._random_state = None

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: Not used
            Kept in the signature for compatibility with parent class.

        classes: None
            Not used by this method.

        sample_weight: None
            Not used by this method.

        Returns
        -------
        self

        """

        row_cnt, _ = X.shape

        if self.samples_seen == 0:
            self._random_state = check_random_state(self.random_state)
            self.n_features = get_dimensions(X)[1]
            self.build_trees()

        for i in range(row_cnt):
            self._partial_fit(X[i])

        return self

    def _partial_fit(self, X):
        """ Train the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.

        """

        # Populates first reference mass profile for every tree defined in the ensemble.
        if self.samples_seen < self.window_size:
            self.update_mass(X, True)
        # Otherwise populates latest mass profile for every tree defined in the ensemble.
        else:
            self.update_mass(X, False)

        if self.is_learning_phase_on is True and self.samples_seen > self.window_size:
            self.set_is_learning_phase_on(False)

        if (self.samples_seen % self.window_size) == 0:
            self.update_models()

        self.samples_seen += 1

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the class labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            y_proba = self.predict_proba(X)
            if y_proba is None:
                # Ensemble is empty, all classes equal, default to zero
                predictions.append(0)
            else:
                # if prediction of this instance is greater than the threshold
                # defined, then this instance is classified as an anomaly.
                if y_proba[0][1] > self.anomaly_threshold:
                    predictions.append(1)
                else:
                    predictions.append(0)
        return np.asarray(predictions)

    def predict_proba(self, X):
        """ Estimate the probability of a sample being normal or abnormal.

        Class probabilities are calculated as the mean predicted class
        probabilities per base estimator.

        Parameters
        ----------
         X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the class probabilities.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for all instances in X.\
            Class probabilities for a sample shall sum to 1 as long as at\
            least one estimators has non-zero predictions. If no estimator can\
            predict probabilities, probabilities of 0 are returned.
        """
        if not self.ensemble:
            # The ensemble is empty, return default predictions (0., 0.)
            return np.zeros((X.shape[0], 2))
        y_proba_mean = None
        max_score = self.window_size * pow(2.0, self.depth)

        for i in range(self.n_estimators):
            y_proba = self.ensemble[i].predict_proba(X, max_score)
            if y_proba_mean is None:
                y_proba_mean = y_proba
            else:
                y_proba_mean = y_proba_mean + (y_proba - y_proba_mean) / (i + 1)
        return normalize(y_proba_mean, norm='l1')

    def initialise_work_space(self):
        """ Initialises work spaces.

        For every dimension in the feature space, creates a minimum and a
        maximum work range.
        """
        for i in range(self.n_features):
            sq = self._random_state.uniform(0, 1)
            min_element = sq - 2 * max(sq, 1 - sq)
            max_element = sq + 2 * max(sq, 1 - sq)
            try:
                self.min_values[i] = min_element
            except IndexError:
                self.min_values.append(min_element)
            try:
                self.max_values[i] = max_element
            except IndexError:
                self.max_values.append(max_element)

    def build_trees(self):
        """
        Initialises ensemble.
        """
        for i in range(self.n_estimators):
            self.initialise_work_space()
            tree = HalfSpaceTree(self.depth, self.n_features, self.size_limit, self.min_values,
                                 self.max_values, self._random_state)
            self.ensemble.append(tree)

    def update_mass(self, X, boolean):
        """
        Populates mass profiles for every tree defined in the ensemble.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.

        boolean: boolean, True or False
            True to update reference mass, False to update latest mass
        """
        for i in range(self.n_estimators):
            self.ensemble[i].update_mass(X, boolean)

    def update_models(self):
        """
        Updates the mass profile of every tree in the ensemble.

        """
        for i in range(self.n_estimators):
            tree = self.ensemble[i]
            tree.update_model(tree.root)

    def set_is_learning_phase_on(self, boolean):
        """
        Sets learning phase in each tree defined in the ensemble.

        Parameters
        ----------
        boolean: Boolean

        """
        self.is_learning_phase_on = boolean
        for i in range(self.n_estimators):
            self.ensemble[i].is_learning_phase_on = boolean


class HalfSpaceTree:

    def __init__(self, max_depth, n_features, size_limit, min_values, max_values,
                 random_state=None):
        """
        Half Space Tree

        Parameters
        ----------
        n_features: int, required
            The dimensionality of the stream.

        max_depth: int, optional (default=15)
            The maximum depth of the trees in the ensemble.
            'maxDepth' in the original paper.

        size_limit: int, optional (default=50)
            The minimum mass required in a node (as a fraction of the window\
            size) to calculate the anomaly score. 'sizeLimit' in the original\
            paper.

        min_values: Array of floats
            Minimum work range for every dimension.

        max_values: Array of floats
            Maximum work range for every dimension.

        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        """
        super().__init__()
        self.max_depth = max_depth
        self.n_features = n_features
        self.size_limit = size_limit
        self.is_learning_phase_on = True
        self.random_state = random_state
        self._random_state = check_random_state(random_state)
        self.root = self.build_tree(min_values, max_values)

    def predict_proba(self, X, max_score):
        """ Predict probabilities of all label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        max_score: float
            The maximum score of an instance could have in the tree.

        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = copy.deepcopy(self.get_votes_for_instance(X[i], max_score))
            y_proba = np.zeros(int(max(votes.keys())) + 1)
            for key, value in votes.items():
                y_proba[int(key)] = value
            predictions.append(y_proba)
        return np.array(predictions)

    def get_votes_for_instance(self, X, max_score):
        """ Get class votes for a single instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        max_score: float
            The maximum score of an instance could have in the tree.
        Returns
        -------
        dict (class_value, weight)

        """
        if self.root is not None and not self.is_learning_phase_on:
            score = self.anomaly_score(X)
            anomaly_score = 1 - score / max_score
            return {0: 1 - anomaly_score, 1: anomaly_score}
        else:
            return {0: 0.5, 1: 0.5}

    def anomaly_score(self, X):
        """
        Calculates anomaly score of a given instance X.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.

        Returns
        -------

        """
        return self._anomaly_score(self.root, X)

    def _anomaly_score(self, node, X):
        """
        Private function to calculate anomaly score.

        Parameters
        ----------
        node: HalfSpaceTreeNode
            Current node

        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.

        Returns
        -------
        anomaly_score: float
            Anomaly score of the instance X.

        """
        if not node.internal_node or node.right_mass <= self.size_limit:
            return node.right_mass * pow(2.0, node.depth)
        else:
            if X[node.split_attribute] > node.split_value:
                return self._anomaly_score(node.right, X)
            else:
                return self._anomaly_score(node.left, X)

    def build_tree(self, min_values, max_values, current_depth=1):
        """
        Builds a single half space tree.

        Parameters
        ----------
        min_values: Array of floats
            Minimum work range for every dimension.
        max_values: Array of floats
            Maximum work range for every dimension.
        current_depth:
            Indicates the current depth of the tree.
        Returns
        -------
        HalfSpaceTreeNode:
            root of the tree.
        """
        if self.max_depth == current_depth:
            return HalfSpaceTreeNode(depth=current_depth, internal_node=False)
        else:
            random_feature_idx = self._random_state.randint(0, self.n_features - 1)
            p = (min_values[random_feature_idx] + max_values[random_feature_idx]) / 2.0
            temp = max_values[random_feature_idx]
            max_values[random_feature_idx] = p
            left = self.build_tree(min_values, max_values, current_depth + 1)
            max_values[random_feature_idx] = temp
            min_values[random_feature_idx] = p
            right = self.build_tree(min_values, max_values, current_depth + 1)
            return HalfSpaceTreeNode(left=left, right=right, split_attribute=random_feature_idx,
                                     split_value=p, depth=current_depth)

    def update_mass(self, X, is_reference_window):
        """
        Records mass data in the tree.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        is_reference_window: Boolean
            True if the classifier is in the first reference window, otherwise false
        Returns
        -------

        """
        return self._update_mass(X, self.root, is_reference_window)

    def _update_mass(self, X, node, is_reference_window):
        """
        Private function where actual mass data is recorded.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        node: HalfSpaceTreeNode

        is_reference_window: Boolean
            True if the classifier is in the first reference window, otherwise false

        """
        if node is None:
            return

        if is_reference_window:
            node.right_mass = node.right_mass + 1
        else:
            node.left_mass = node.left_mass + 1

        if node.depth < self.max_depth:
            if X[node.split_attribute] > node.split_value:
                self._update_mass(X, node.right, is_reference_window)
            else:
                self._update_mass(X, node.left, is_reference_window)
        else:
            return

    def update_model(self, node):
        """
        Updates model to adapt to new data.

        Parameters
        ----------
        node: HalfSpaceTreeNode

        """
        if node is not None:
            if node.right_mass != 0 or node.left_mass != 0:
                node.right_mass = node.left_mass

            if node.left_mass != 0:
                node.left_mass = 0

            self.update_model(node.left)
            self.update_model(node.right)
        return


class HalfSpaceTreeNode:

    def __init__(self, left=None, right=None, split_value=0.0, split_attribute=-1,
                 depth=1, internal_node=True):
        """
        Parameters
        ----------
        left: HalfSpaceTreeNode, optional (default=None)
            The left node pointed to this node.

        right: HalfSpaceTreeNode, optional (default=None)
            The left node pointed to this node.

        split_value: double, optional (default=0.0)
            Mid point of the selected random dimension.
            'p' in the original paper.

        split_attribute: int, optional (default=-1)
            Randomly selected attribute for the node.
            'q' in the original paper.

        depth: int, optional (default=1)
            Represents depth of the node in the tree.
            'k' in the original paper.

        internal_node: boolean, optional (default=True)
            True if node is an internal node, false if it is an leaf node.

        Attributes
        ----------
        left_mass: int
            Mass of the node in the reference window.

        right_mass: int
            Mass of the node in the latest window.

        """
        self.left = left
        self.right = right
        self.left_mass = 0
        self.right_mass = 0
        self.depth = depth
        self.split_value = split_value
        self.split_attribute = split_attribute
        self.internal_node = internal_node
