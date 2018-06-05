from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
import numpy as np
import operator
from skmultiflow.core.utils.utils import *


class BrHoeffdingTree:
    """ Binary Relevance Hoeffding Tree

    Binary Relevance (BR)-based methods are composed of binary classifiers; one
    for each label.
    This classifiers are Hoeffding trees, trained separately and no correlation
    between the labels in taken into account.


    """
    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_features=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 categorical_features=None):
        """Binary Relevance HoeffdingTree class constructor."""
        super().__init__()
        self.max_byte_size = max_byte_size
        self.memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_features = remove_poor_features
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.categorical_features = categorical_features

        self.hoeffding_trees = []

    def fit(self, X, Y, classes=None):
        raise NotImplementedError

    def partial_fit(self, X, Y, classes=None, weight=None):
        """Incrementally trains the model. Train samples (instances) are composed of X features and their
        corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for the instance and update the leaf node
          statistics.
        * If growth is allowed and the number of instances that the leaf has observed between split attempts
          exceed the grace period then attempt to split.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: Not used.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        if len(self.hoeffding_trees) == 0:
            self.n_labels = len(Y[0])
            self.hoeffding_trees = [HoeffdingTree(self.max_byte_size,
                                                  self.memory_estimate_period,
                                                  self.grace_period,
                                                  self.split_criterion,
                                                  self.split_confidence,
                                                  self.tie_threshold,
                                                  self.binary_split,
                                                  self.stop_mem_management,
                                                  self.remove_poor_features,
                                                  self.no_preprune,
                                                  self.leaf_prediction,
                                                  self.nb_threshold,
                                                  self.categorical_features)
                                    for i in range(self.n_labels)]
        for label in range(self.n_labels):
            self.hoeffding_trees[label].partial_fit(X, Y[:, label], classes, weight)

    def predict_proba(self, X):
        """Predicts probabilities of all label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        list
            Predicted the probabilities of all the labels for all instances in X.

        """
        if len(self.hoeffding_trees) == 0:
            y_proba = np.zeros((len(X), self.n_labels))
        else:
            y_proba = []
            for index in range(len(X)):
                label_proba = []
                for label in range(self.n_labels):
                    label_proba.append(self.hoeffding_trees[label].predict_proba(np.array([list(X[index])]))[0])
                y_proba.append(label_proba)
        return y_proba

    def predict(self, X):
        """Predicts the label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        list
            Predicted labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            pred = []
            for label in range(self.n_labels):
                index, _ = max(enumerate(y_proba[i][label]), key=operator.itemgetter(1))
                pred.append(index)
            predictions.append(pred)
        return predictions







