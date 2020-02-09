import numpy as np

from skmultiflow.utils.data_structures import FastBuffer, FastComplexBuffer, ConfusionMatrix, MOLConfusionMatrix
from skmultiflow.utils import check_weights

from timeit import default_timer as timer

import warnings


class ClassificationMeasurements(object):
    """ Classification measurements.

    Class used to keep updated statistics about a classifier, in order
    to be able to provide, at any given moment, any relevant metric about
    that classifier.

    It combines a ConfusionMatrix object, with some additional statistics,
    to compute a range of performance metrics. Important: indices in the
    confusion matrix depend on the arrival order of observed classes.

    In order to keep statistics updated, the class won't require lots of
    information, but two: the predictions and true labels.

    At any given moment, it can compute the multiple statistics including
    accuracy, kappa, kappa_t, kappa_m, majority_class, error rate, etc.

    Parameters
    ----------
    targets: list
        A list containing the possible labels.

    dtype: data type (Default: numpy.int64)
        The data type of the existing labels.

    Examples
    --------

    """

    def __init__(self, targets=None, dtype=np.int64):
        warnings.warn("'{}' will be removed in v0.7. Use 'ClassificationPerformanceEvaluator' instead.".
                      format(type(self).__name__), category=FutureWarning)
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
            self.targets = targets
        else:
            self.n_targets = 2
            self.targets = [0, 1]
        self.confusion_matrix = ConfusionMatrix(self.n_targets, dtype)
        self.last_true_label = None
        self.last_prediction = None
        self.last_sample = None
        self.sample_count = 0
        self.majority_classifier = 0
        self.correct_no_change = 0

    def reset(self):
        if self.targets is not None:
            self.n_targets = len(self.targets)
        else:
            self.n_targets = 2
        self.last_true_label = None
        self.last_prediction = None
        self.last_sample = None
        self.sample_count = 0
        self.majority_classifier = 0
        self.correct_no_change = 0
        self.confusion_matrix.restart(self.n_targets)

    def add_result(self, y_true, y_pred, weight=1.0):
        """ Updates its statistics with the results of a prediction.

        Parameters
        ----------
        y_true: int
            The true label.

        y_pred: int
            The classifier's prediction

        weight: float
            Sample's weight

        """
        check_weights(weight)

        true_y = self._get_target_index(y_true, True)
        pred = self._get_target_index(y_pred, True)
        self.confusion_matrix.update(true_y, pred)
        self.sample_count += 1

        if self.get_majority_class() == y_true:
            self.majority_classifier += weight
        if self.last_true_label == y_true:
            self.correct_no_change += weight

        self.last_true_label = y_true
        self.last_prediction = y_pred

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_majority_class(self):
        """ Computes the true majority class.

        Returns
        -------
        int
            The true majority class.

        """
        if (self.n_targets is None) or (self.n_targets == 0):
            return False
        majority_idx = 0
        max_prob = 0.0
        for i in range(self.n_targets):
            sum_value = 0.0
            for j in range(self.n_targets):
                sum_value += self.confusion_matrix.value_at(i, j)
            sum_value = sum_value / self.sample_count
            if sum_value > max_prob:
                max_prob = sum_value
                majority_idx = i
        return self.targets[majority_idx]

    def get_accuracy(self):
        """ Computes the accuracy.

        Returns
        -------
        float
            The accuracy.

        """
        sum_value = 0.0
        n, _ = self.confusion_matrix.shape()
        for i in range(n):
            sum_value += self.confusion_matrix.value_at(i, i)
        try:
            return sum_value / self.sample_count
        except ZeroDivisionError:
            return 0.0

    def get_incorrectly_classified_ratio(self):
        return 1.0 - self.get_accuracy()

    def _get_target_index(self, target, add_label=False):
        """ Computes the index of an element in the self.targets list.
        Also reshapes the ConfusionMatrix and adds new found targets
        if add is True.

        Parameters
        ----------
        target: int
            A class label.

        add_label: bool
            Either to add new found labels to the targets list or not.

        Returns
        -------
        int
            The target index in the self.targets list.

        """
        if (self.targets is None) and add_label:
            self.targets = []
            self.targets.append(target)
            self.n_targets = len(self.targets)
            if self.n_targets > 2:
                # The default matrix is for the binary case, extend it if necessary
                self.confusion_matrix.reshape(self.n_targets, self.n_targets)
        elif (self.targets is None) and (not add_label):
            return None
        if (target not in self.targets) and add_label:
            self.targets.append(target)
            self.n_targets = len(self.targets)
            if self.confusion_matrix.shape()[0] < self.n_targets:
                self.confusion_matrix.reshape(self.n_targets, self.n_targets)
        for i in range(len(self.targets)):
            if self.targets[i] == target:
                return i
        return None

    def get_kappa(self):
        """ Computes the Cohen's kappa coefficient.

        Returns
        -------
        float
            The Cohen's kappa coefficient.

        """
        p0 = self.get_accuracy()
        pc = 0.0
        n_rows, n_cols = self.confusion_matrix.shape()
        for i in range(n_rows):
            row = self.confusion_matrix.row(i)
            column = self.confusion_matrix.column(i)

            sum_row = np.sum(row) / self.sample_count
            sum_column = np.sum(column) / self.sample_count

            pc += sum_row * sum_column
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_t(self):
        """ Computes the Cohen's kappa T coefficient. This measures the
        temporal correlation between samples.

        Returns
        -------
        float
            The Cohen's kappa T coefficient.

        """
        p0 = self.get_accuracy()
        if self.sample_count != 0:
            pc = self.correct_no_change / self.sample_count
        else:
            pc = 0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_m(self):
        """ Computes the Cohen's kappa M coefficient.

        Returns
        -------
        float
            The Cohen's kappa M coefficient.

        """
        p0 = self.get_accuracy()
        if self.sample_count != 0:
            pc = self.majority_classifier / self.sample_count
        else:
            pc = 0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_g_mean(self):
        """ Compute the G-mean of the classifier. Binary-classification only.

        Returns
        -------
        float
            The G-mean
        """
        negative_idx = self._get_target_index(0)
        if negative_idx is None:
            return 0.0
        else:
            tn = self.confusion_matrix.value_at(negative_idx, negative_idx)
            fp = self.confusion_matrix.value_at(negative_idx, 1-negative_idx)
            if tn + fp == 0:
                specificity = 0
            else:
                specificity = tn / (tn + fp)
            sensitivity = self.get_recall()
            return np.sqrt((sensitivity * specificity))

    def get_f1_score(self):
        """ Compute the F1-score of the classifier. Binary-classification only.

        Returns
        -------
        float
            The F1-score
        """
        precision = self.get_precision()
        recall = self.get_recall()
        if recall + precision == 0:
            return 0.0
        else:
            return 2 * ((precision * recall) / (precision + recall))

    def get_precision(self):
        """ compute the precision of the classifier. Binary-classification only.


        Returns
        -------
        float
            The precision
        """
        positive_idx = self._get_target_index(1)
        if positive_idx is None:
            return 0.0
        else:
            tp = self.confusion_matrix.value_at(positive_idx, positive_idx)
            fp = self.confusion_matrix.value_at(1-positive_idx, positive_idx)
            if tp + fp == 0:
                return 0.0
            else:
                return tp / (tp + fp)

    def get_recall(self):
        """ Compute the recall of the classifier. Binary-classification only.

        Returns
        -------
        float
            The recall.
        """
        positive_idx = self._get_target_index(1)
        if positive_idx is None:
            return 0.0
        else:
            tp = self.confusion_matrix.value_at(positive_idx, positive_idx)
            fn = self.confusion_matrix.value_at(positive_idx, 1-positive_idx)
            if tp + fn == 0:
                return 0.0
            else:
                return tp / (tp + fn)

    @property
    def _matrix(self):
        return self.confusion_matrix.matrix

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               ' - sample_count: {}'.format(self.sample_count) + \
               ' - accuracy: {:.6f}'.format(self.get_accuracy()) + \
               ' - kappa: {:.6f}'.format(self.get_kappa()) + \
               ' - kappa_t: {:.6f}'.format(self.get_kappa_t()) + \
               ' - kappa_m: {:.6f}'.format(self.get_kappa_m()) + \
               ' - f1-score: {:.6f}'.format(self.get_f1_score()) + \
               ' - precision: {:.6f}'.format(self.get_precision()) + \
               ' - recall: {:.6f}'.format(self.get_recall()) + \
               ' - g-mean: {:.6f}'.format(self.get_g_mean()) + \
               ' - majority_class: {}'.format(self.get_majority_class())


class WindowClassificationMeasurements(object):
    """ This class will maintain a fixed sized window of the newest information
    about one classifier. It can provide, as requested, any of the relevant
    current metrics about the classifier, measured inside the window.

    To keep track of statistics inside a window, the class will use a
    ConfusionMatrix object, alongside FastBuffers, to simulate fixed sized
    windows of the important classifier's attributes.

    Its functionality is somewhat similar to those of the
    ClassificationMeasurements class. The difference is that the statistics
    kept by this class are local, or partial, while the statistics kept by
    the ClassificationMeasurements class are global.

    At any given moment, it can compute multiple statistics including accuracy,
    kappa, kappa_t, kappa_m, majority_class, error rate, etc.

    Parameters
    ----------
    targets: list
        A list containing the possible labels.

    dtype: data type (Default: numpy.int64)
        The data type of the existing labels.

    window_size: int (Default: 200)
        The width of the window. Determines how many samples the object
        can see.

    Examples
    --------

    """

    def __init__(self, targets=None, dtype=np.int64, window_size=200):
        warnings.warn("'{}' will be removed in v0.7. Use 'WindowClassificationPerformanceEvaluator' instead.".
                      format(type(self).__name__), category=FutureWarning)
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
            self.targets = targets
        else:
            self.n_targets = 2
            self.targets = [0, 1]
        self.confusion_matrix = ConfusionMatrix(self.n_targets, dtype)
        self.last_class = None

        self.window_size = window_size
        self.true_labels = FastBuffer(window_size)
        self.predictions = FastBuffer(window_size)
        self.temp = 0
        self.last_prediction = None
        self.last_true_label = None
        self.last_sample = None

        self.majority_classifier = 0
        self.correct_no_change = 0
        self.majority_classifier_correction = FastBuffer(window_size)
        self.correct_no_change_correction = FastBuffer(window_size)

    def reset(self):
        if self.targets is not None:
            self.n_targets = len(self.targets)
        else:
            self.n_targets = 2

        self.true_labels = FastBuffer(self.window_size)
        self.predictions = FastBuffer(self.window_size)
        self.temp = 0
        self.last_prediction = None
        self.last_true_label = None
        self.last_sample = None

        self.majority_classifier = 0
        self.correct_no_change = 0
        self.confusion_matrix.restart(self.n_targets)
        self.majority_classifier_correction = FastBuffer(self.window_size)
        self.correct_no_change_correction = FastBuffer(self.window_size)

    def add_result(self, y_true, y_pred, weight=1.0):
        """ Updates its statistics with the results of a prediction.
        If needed it will remove samples from the observation window.

        Parameters
        ----------
        y_true: int
            The true label.

        y_pred: int
            The classifier's prediction

        weight: float
            Sample's weight
        """
        check_weights(weight)
        true_y = self._get_target_index(y_true, True)
        pred = self._get_target_index(y_pred, True)
        old_true = self.true_labels.add_element(np.array([y_true]))
        old_predict = self.predictions.add_element(np.array([y_pred]))

        # Verify if it's needed to decrease the count of any label
        # pair in the confusion matrix
        if (old_true is not None) and (old_predict is not None):
            self.temp += 1
            self.confusion_matrix.remove(self._get_target_index(old_true[0]),
                                         self._get_target_index(old_predict[0]))
            self.correct_no_change += self.correct_no_change_correction.peek()
            self.majority_classifier += self.majority_classifier_correction.peek()

        # Verify if it's needed to decrease the majority_classifier count
        majority_class = self.get_majority_class()
        if (majority_class == y_true) and (majority_class is not None):
            self.majority_classifier += weight
            self.majority_classifier_correction.add_element([-1])
        else:
            self.majority_classifier_correction.add_element([0])

        # Verify if it's needed to decrease the correct_no_change
        if (self.last_true_label == y_true) and (self.last_true_label is not None):
            self.correct_no_change += weight
            self.correct_no_change_correction.add_element([-1])
        else:
            self.correct_no_change_correction.add_element([0])

        self.confusion_matrix.update(true_y, pred, weight=weight)

        self.last_true_label = y_true
        self.last_prediction = y_pred

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_majority_class(self):
        """ Computes the window/current true majority class.

        Returns
        -------
        int
            The true window/current majority class.

        """
        if (self.n_targets is None) or (self.n_targets == 0):
            return None
        majority_idx = 0
        max_prob = 0.0
        for i in range(self.n_targets):
            sum_value = 0.0
            for j in range(self.n_targets):
                sum_value += self.confusion_matrix.value_at(i, j)
            sum_value = sum_value / self.true_labels.get_current_size()
            if sum_value > max_prob:
                max_prob = sum_value
                majority_idx = i
        return self.targets[majority_idx]

    def get_accuracy(self):
        """ Computes the window/current accuracy.

        Returns
        -------
        float
            The window/current accuracy.

        """
        sum_value = 0.0
        n, _ = self.confusion_matrix.shape()
        for i in range(n):
            sum_value += self.confusion_matrix.value_at(i, i)
        try:
            return sum_value / self.true_labels.get_current_size()
        except ZeroDivisionError:
            return 0.0

    def get_incorrectly_classified_ratio(self):
        return 1.0 - self.get_accuracy()

    def _get_target_index(self, target, add=False):
        """ Computes the index of an element in the self.targets list.
        Also reshapes the ConfusionMatrix and adds new found targets
        if add is True.

        Parameters
        ----------
        target: int
            A class label.

        add: bool
            Either to add new found labels to the targets list or not.

        Returns
        -------
        int
            The target index in the self.targets list.

        """
        if (self.targets is None) and add:
            self.targets = []
            self.targets.append(target)
            self.n_targets = len(self.targets)
            if self.n_targets > 2:
                # The default matrix is for the binary case, extend it if necessary
                self.confusion_matrix.reshape(self.n_targets, self.n_targets)
        elif (self.targets is None) and (not add):
            return None
        if target not in self.targets and add:
            self.targets.append(target)
            self.n_targets = len(self.targets)
            if self.confusion_matrix.shape()[0] < self.n_targets:
                self.confusion_matrix.reshape(self.n_targets, self.n_targets)
        for i in range(len(self.targets)):
            if self.targets[i] == target:
                return i
        return None

    def get_kappa(self):
        """ Computes the window/current Cohen's kappa coefficient.

        Returns
        -------
        float
            The window/current Cohen's kappa coefficient.

        """
        p0 = self.get_accuracy()
        pc = 0.0
        n_rows, n_cols = self.confusion_matrix.shape()
        for i in range(n_rows):
            row = self.confusion_matrix.row(i)
            column = self.confusion_matrix.column(i)

            sum_row = np.sum(row) / self.true_labels.get_current_size()
            sum_column = np.sum(column) / self.true_labels.get_current_size()

            pc += sum_row * sum_column

        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_t(self):
        """ Computes the window/current Cohen's kappa T coefficient. This measures
        the temporal correlation between samples.

        Returns
        -------
        float
            The window/current Cohen's kappa T coefficient.

        """
        p0 = self.get_accuracy()
        if self.sample_count != 0:
            pc = self.correct_no_change / self.sample_count
        else:
            pc = 0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_kappa_m(self):
        """ Computes the window/current Cohen's kappa M coefficient.

        Returns
        -------
        float
            The window/current Cohen's kappa M coefficient.

        """
        p0 = self.get_accuracy()
        if self.sample_count != 0:
            pc = self.majority_classifier / self.sample_count
        else:
            pc = 0
        if pc == 1:
            return 1
        return (p0 - pc) / (1.0 - pc)

    def get_g_mean(self):
        """ Compute the G-mean of the classifier. Binary-classification only.

        Returns
        -------
        float
            The G-mean
        """
        negative_idx = self._get_target_index(0)
        if negative_idx is None:
            return 0.0
        else:
            tn = self.confusion_matrix.value_at(negative_idx, negative_idx)
            fp = self.confusion_matrix.value_at(negative_idx, 1 - negative_idx)
            if tn + fp == 0:
                specificity = 0
            else:
                specificity = tn / (tn + fp)
            sensitivity = self.get_recall()
            return np.sqrt((sensitivity * specificity))

    def get_f1_score(self):
        """ Compute the F1-score of the classifier. Binary-classification only.

        Returns
        -------
        float
            The F1-score
        """
        precision = self.get_precision()
        recall = self.get_recall()
        if recall + precision == 0:
            return 0.0
        else:
            return 2 * ((precision * recall) / (precision + recall))

    def get_precision(self):
        """ compute the precision of the classifier. Binary-classification only.


        Returns
        -------
        float
            The precision
        """
        positive_idx = self._get_target_index(1)
        if positive_idx is None:
            return 0.0
        else:
            tp = self.confusion_matrix.value_at(positive_idx, positive_idx)
            fp = self.confusion_matrix.value_at(1 - positive_idx, positive_idx)
            if tp + fp == 0:
                return 0.0
            else:
                return tp / (tp + fp)

    def get_recall(self):
        """ Compute the recall of the classifier. Binary-classification only.

        Returns
        -------
        float
            The recall.
        """
        positive_idx = self._get_target_index(1)
        if positive_idx is None:
            return 0.0
        else:
            tp = self.confusion_matrix.value_at(positive_idx, positive_idx)
            fn = self.confusion_matrix.value_at(positive_idx, 1 - positive_idx)
            if tp + fn == 0:
                return 0.0
            else:
                return tp / (tp + fn)

    @property
    def _matrix(self):
        return self.confusion_matrix.matrix

    @property
    def sample_count(self):
        return self.true_labels.get_current_size()

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               ' - sample_count: {}'.format(self.sample_count) + \
               ' - window_size: {}'.format(self.window_size) + \
               ' - accuracy: {:.6f}'.format(self.get_accuracy()) + \
               ' - kappa: {:.6f}'.format(self.get_kappa()) + \
               ' - kappa_t: {:.6f}'.format(self.get_kappa_t()) + \
               ' - kappa_m: {:.6f}'.format(self.get_kappa_m()) + \
               ' - f1-score: {:.6f}'.format(self.get_f1_score()) + \
               ' - precision: {:.6f}'.format(self.get_precision()) + \
               ' - recall: {:.6f}'.format(self.get_recall()) + \
               ' - g-mean: {:.6f}'.format(self.get_g_mean()) + \
               ' - majority_class: {}'.format(self.get_majority_class())


class MultiTargetClassificationMeasurements(object):
    """ This class will keep updated statistics about a multi output classifier,
    using a confusion matrix adapted to multi output problems, the
    MOLConfusionMatrix, alongside other relevant attributes.

    The performance metrics for multi output tasks are different from those used
    for normal classification tasks. Thus, the statistics provided by this class
    are different from those provided by the ClassificationMeasurements and from
    the WindowClassificationMeasurements.

    At any given moment, it can compute the following statistics: hamming_loss,
    hamming_score, exact_match and j_index.

    Parameters
    ----------
    targets: list
        A list containing the possible labels.

    dtype: data type (Default: numpy.int64)
        The data type of the existing labels.

    Examples
    --------

    """

    def __init__(self, targets=None, dtype=np.int64):
        warnings.warn("'{}' will be removed in v0.7. Use 'MultiLabelClassificationPerformanceEvaluator' instead.".
                      format(type(self).__name__), category=FutureWarning)
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = MOLConfusionMatrix(self.n_targets, dtype)
        self.last_true_label = None
        self.last_prediction = None
        self.sample_count = 0
        self.targets = targets
        self.exact_match_count = 0
        self.j_sum = 0

    def reset(self):
        if self.targets is not None:
            self.n_targets = len(self.targets)
        else:
            self.n_targets = 0
        self.confusion_matrix.restart(self.n_targets)
        self.last_true_label = None
        self.last_prediction = None
        self.sample_count = 0
        self.exact_match_count = 0
        self.j_sum = 0

    def add_result(self, y_true, y_pred):
        """ Updates its statistics with the results of a prediction.

        Adds the result to the MOLConfusionMatrix and update exact_matches and
        j-index sum counts.

        Parameters
        ----------
        y_true: list or numpy.ndarray
            The true label.

        y_pred: list or numpy.ndarray
            The classifier's prediction

        """

        self.last_true_label = y_true
        self.last_prediction = y_pred
        m = 0
        if isinstance(y_true, np.ndarray):
            m = y_true.size
        elif isinstance(y_true, list):
            m = len(y_true)
        self.n_targets = m
        equal = True
        for i in range(m):
            self.confusion_matrix.update(i, y_true[i], y_pred[i])
            # update exact_match count
            if y_true[i] != y_pred[i]:
                equal = False

        # update exact_match
        if equal:
            self.exact_match_count += 1

        # update j_index count
        inter = sum((y_true * y_pred) > 0) * 1.
        union = sum((y_true + y_pred) > 0) * 1.
        if union > 0:
            self.j_sum += inter / union
        elif np.sum(y_true) == 0:
            self.j_sum += 1

        self.sample_count += 1

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_hamming_loss(self):
        """ Computes the Hamming loss, which is the complement of the
        Hamming score metric.

        Returns
        -------
        float
            The hamming loss.

        """
        return 1.0 - self.get_hamming_score()

    def get_hamming_score(self):
        """ Computes the Hamming score, defined as the number of correctly
        classified labels divided by the total number of labels classified.

        Returns
        -------
        float
            The Hamming score.

        """
        try:
            return self.confusion_matrix.get_sum_main_diagonal() / (self.sample_count * self.n_targets)
        except ZeroDivisionError:
            return 0.0

    def get_exact_match(self):
        """ Computes the exact match metric.

        This is the most strict multi output metric, defined as the number of
        samples that have all their labels correctly classified, divided by the
        total number of samples.

        Returns
        -------
        float
            The exact match metric.

        """
        return self.exact_match_count / self.sample_count

    def get_j_index(self):
        """ Computes the Jaccard index, also known as the intersection over union
        metric. It is calculated by dividing the number of correctly classified
        labels by the union of predicted and true labels.

        Returns
        -------
        float
            The Jaccard index.

        """
        return self.j_sum / self.sample_count

    def get_total_sum(self):
        return self.confusion_matrix.get_total_sum()

    @property
    def _matrix(self):
        return self.confusion_matrix.matrix

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               ' - sample_count: {}'.format(self.sample_count) + \
               ' - hamming_loss: {:.6f}'.format(self.get_hamming_loss()) + \
               ' - hamming_score: {:.6f}'.format(self.get_hamming_score()) + \
               ' - exact_match: {:.6f}'.format(self.get_exact_match()) + \
               ' - j_index: {:.6f}'.format(self.get_j_index())


class WindowMultiTargetClassificationMeasurements(object):
    """ This class will maintain a fixed sized window of the newest information
    about one classifier. It can provide, as requested, any of the relevant
    current metrics about the classifier, measured inside the window.

    This class will keep updated statistics about a multi output classifier,
    using a confusion matrix adapted to multi output problems, the
    MOLConfusionMatrix, alongside other of the classifier's relevant
    attributes stored in ComplexFastBuffer objects, which will simulate
    fixed sized windows.

    Its functionality is somewhat similar to those of the
    MultiTargetClassificationMeasurements class. The difference is that the statistics
    kept by this class are local, or partial, while the statistics kept by
    the MultiTargetClassificationMeasurements class are global.

    At any given moment, it can compute the following statistics: hamming_loss,
    hamming_score, exact_match and j_index.

    Parameters
    ----------
    targets: list
        A list containing the possible labels.

    dtype: data type (Default: numpy.int64)
        The data type of the existing labels.

    window_size: int (Default: 200)
        The width of the window. Determines how many samples the object
        can see.

    Examples
    --------

    """

    def __init__(self, targets=None, dtype=np.int64, window_size=200):
        warnings.warn("'{}' will be removed in v0.7. Use 'WindowMultiLabelClassificationPerformanceEvaluator' instead.".
                      format(type(self).__name__), category=FutureWarning)
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = MOLConfusionMatrix(self.n_targets, dtype)
        self.last_true_label = None
        self.last_prediction = None

        self.targets = targets
        self.window_size = window_size
        self.exact_match_count = 0
        self.j_sum = 0
        self.true_labels = FastComplexBuffer(window_size, self.n_targets)
        self.predictions = FastComplexBuffer(window_size, self.n_targets)

    def reset(self):
        if self.targets is not None:
            self.n_targets = len(self.targets)
        else:
            self.n_targets = 0
        self.confusion_matrix.restart(self.n_targets)
        self.last_true_label = None
        self.last_prediction = None
        self.exact_match_count = 0
        self.j_sum = 0
        self.true_labels = FastComplexBuffer(self.window_size, self.n_targets)
        self.predictions = FastComplexBuffer(self.window_size, self.n_targets)

    def add_result(self, y_true, y_pred):
        """ Updates its statistics with the results of a prediction.

        Adds the result to the MOLConfusionMatrix, and updates the
        ComplexFastBuffer objects.

        Parameters
        ----------
        y_true: list or numpy.ndarray
            The true label.

        y_pred: list or numpy.ndarray
            The classifier's prediction

        """

        self.last_true_label = y_true
        self.last_prediction = y_pred
        m = 0
        if hasattr(y_true, 'size'):
            m = y_true.size
        elif hasattr(y_true, 'append'):
            m = len(y_true)
        self.n_targets = m

        for i in range(m):
            self.confusion_matrix.update(i, y_true[i], y_pred[i])

        old_true = self.true_labels.add_element(y_true)
        old_predict = self.predictions.add_element(y_pred)
        if (old_true is not None) and (old_predict is not None):
            for i in range(m):
                self.confusion_matrix.remove(old_true[0][i], old_predict[0][i])

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_hamming_loss(self):
        """ Computes the window/current Hamming loss, which is the
        complement of the Hamming score metric.

        Returns
        -------
        float
            The window/current hamming loss.

        """
        return 1.0 - self.get_hamming_score()

    def get_hamming_score(self):
        """ Computes the window/current Hamming score, defined as the number of
        correctly classified labels divided by the total number of labels
        classified.

        Returns
        -------
        float
            The window/current hamming score.

        """
        return hamming_score(self.true_labels.get_queue(), self.predictions.get_queue())

    def get_exact_match(self):
        """ Computes the window/current exact match metric.

        This is the most strict multi output metric, defined as the number of
        samples that have all their labels correctly classified, divided by the
        total number of samples.

        Returns
        -------
        float
            The window/current exact match metric.

        """
        return exact_match(self.true_labels.get_queue(), self.predictions.get_queue())

    def get_j_index(self):
        """ Computes the window/current Jaccard index, also known as the intersection
        over union metric. It is calculated by dividing the number of correctly
        classified labels by the union of predicted and true labels.

        Returns
        -------
        float
            The window/current Jaccard index.

        """
        return j_index(self.true_labels.get_queue(), self.predictions.get_queue())

    def get_total_sum(self):
        return self.confusion_matrix.get_total_sum()

    @property
    def matrix(self):
        return self.confusion_matrix.matrix

    @property
    def sample_count(self):
        return self.true_labels.get_current_size()

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               ' - sample_count: {}'.format(self.sample_count) + \
               ' - hamming_loss: {:.6f}'.format(self.get_hamming_loss()) + \
               ' - hamming_score: {:.6f}'.format(self.get_hamming_score()) + \
               ' - exact_match: {:.6f}'.format(self.get_exact_match()) + \
               ' - j_index: {:.6f}'.format(self.get_j_index())


class RegressionMeasurements(object):
    """ This class is used to keep updated statistics over a regression
    learner in a regression problem context.

    It will keep track of global metrics, that can be provided at
    any moment. The relevant metrics kept by an instance of this class
    are: MSE (mean square error) and MAE (mean absolute error).

    """

    def __init__(self):
        super().__init__()
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    def reset(self):
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    def add_result(self, y_true, y_pred):
        """ Use the true value and the prediction to update the statistics.

        Parameters
        ----------
        y_true: float
            The true value.

        y_pred: float
            The predicted value.

        """
        self.last_true_label = y_true
        self.last_prediction = y_pred

        self.total_square_error += (y_true - y_pred) * (y_true - y_pred)
        self.average_error += np.absolute(y_true - y_pred)
        self.sample_count += 1

    def get_mean_square_error(self):
        """ Computes the mean square error.

        Returns
        -------
        float
            The mean square error.

        """
        if self.sample_count == 0:
            return 0.0
        else:
            return self.total_square_error / self.sample_count

    def get_average_error(self):
        """ Computes the average error.

        Returns
        -------
        float
            The average error.

        """
        if self.sample_count == 0:
            return 0.0
        else:
            return self.average_error / self.sample_count

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               ' - sample_count: {}'.format(self.sample_count) + \
               ' - mean_square_error: {:.6f}'.format(self.get_mean_square_error()) + \
               ' - mean_absolute_error: {:.6f}'.format(self.get_average_error())


class WindowRegressionMeasurements(object):
    """ This class is used to keep updated statistics over a regression
    learner in a regression problem context inside a fixed sized window.
    It uses FastBuffer objects to simulate the fixed sized windows.

    It will keep track of partial metrics, that can be provided at
    any moment. The relevant metrics kept by an instance of this class
    are: MSE (mean square error) and MAE (mean absolute error).

    """

    def __init__(self, window_size=200):
        super().__init__()
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.total_square_error_correction = FastBuffer(window_size)
        self.average_error_correction = FastBuffer(window_size)
        self.window_size = window_size

    def reset(self):
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.total_square_error_correction = FastBuffer(self.window_size)
        self.average_error_correction = FastBuffer(self.window_size)

    def add_result(self, y_true, y_pred):
        """ Use the true value and the prediction to update the statistics.

        Parameters
        ----------
        y_true: float
            The true value.

        y_pred: float
            The predicted value.

        """

        self.last_true_label = y_true
        self.last_prediction = y_pred
        self.total_square_error += (y_true - y_pred) * (y_true - y_pred)
        self.average_error += np.absolute(y_true - y_pred)

        old_square = self.total_square_error_correction.add_element(
            np.array([-1 * ((y_true - y_pred) * (y_true - y_pred))]))
        old_average = self.average_error_correction.add_element(np.array([-1 * (np.absolute(y_true - y_pred))]))

        if (old_square is not None) and (old_average is not None):
            self.total_square_error += old_square[0]
            self.average_error += old_average[0]

    def get_mean_square_error(self):
        """ Computes the window/current mean square error.

        Returns
        -------
        float
            The window/current mean square error.

        """
        if self.sample_count == 0:
            return 0.0
        else:
            return self.total_square_error / self.sample_count

    def get_average_error(self):
        """ Computes the window/current average error.

        Returns
        -------
        float
            The window/current average error.

        """
        if self.sample_count == 0:
            return 0.0
        else:
            return self.average_error / self.sample_count

    def get_last(self):
        return self.last_true_label, self.last_prediction

    @property
    def sample_count(self):
        return self.total_square_error_correction.get_current_size()

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               ' - sample_count: {}'.format(self.sample_count) + \
               ' - mean_square_error: {:.6f}'.format(self.get_mean_square_error()) + \
               ' - mean_absolute_error: {:.6f}'.format(self.get_average_error())


class MultiTargetRegressionMeasurements(object):
    """ This class is used to keep updated statistics over a multi-target regression
    learner in a multi-target regression problem context.

    It will keep track of global metrics, that can be provided at
    any moment. The relevant metrics kept by an instance of this class
    are: AMSE (average mean square error), AMAE (average mean absolute error),
    and ARMSE (average root mean square error).
    """

    def __init__(self):
        super().__init__()
        self.n_targets = 0
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    def reset(self):
        self.n_targets = 0
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    def add_result(self, y, prediction):
        """ Use the true value and the prediction to update the statistics.

        Parameters
        ----------
        y: float or list or np.ndarray
            The true value(s).

        prediction: float or list or np.ndarray
            The predicted value(s).

        prediction: float or list or np.ndarray
            The predicted value(s).

        """

        self.last_true_label = y
        self.last_prediction = prediction

        m = 0
        if hasattr(y, 'size'):
            m = y.size
        elif hasattr(y, 'append'):
            m = len(y)
        self.n_targets = m

        self.total_square_error += (y - prediction) * (y - prediction)
        self.average_error += np.absolute(y - prediction)
        self.sample_count += 1

    def get_average_mean_square_error(self):
        """ Computes the average mean square error.

        Returns
        -------
        float
            The average mean square error.
        """
        if self.sample_count == 0:
            return 0.0
        else:
            return np.sum(self.total_square_error / self.sample_count) / self.n_targets

    def get_average_absolute_error(self):
        """ Computes the average mean absolute error.

        Returns
        -------
        float
            The average absolute error.
        """
        if self.sample_count == 0:
            return 0.0
        else:
            return np.sum(self.average_error / self.sample_count) \
                / self.n_targets

    def get_average_root_mean_square_error(self):
        """ Computes the mean square error.

        Returns
        -------
        float
            The average mean square error.
        """
        if self.sample_count == 0:
            return 0.0
        else:
            mse = self.total_square_error / self.sample_count
            return np.sum(
                np.sqrt(mse, out=np.zeros_like(mse), where=mse >= 0.0)
            ) / self.n_targets

    def get_last(self):
        return self.last_true_label, self.last_prediction

    @property
    def _sample_count(self):
        return self.sample_count

    def get_info(self):
        return 'MultiTargetRegressionMeasurements: sample_count: ' + \
                str(self._sample_count) + ' - average_mean_square_error: ' + \
                str(self.get_average_mean_square_error()) + ' - average_mean_absolute_error: ' + \
                str(self.get_average_absolute_error()) + ' - average_root_mean_square_error: ' + \
                str(self.get_average_root_mean_square_error())


class WindowMultiTargetRegressionMeasurements(object):
    """ This class is used to keep updated statistics over a multi-target regression
    learner in a multi-target regression problem context inside a fixed sized
    window. It uses FastBuffer objects to simulate the fixed sized windows.

    It will keep track of partial metrics, that can be provided at
    any moment. The relevant metrics kept by an instance of this class
    are: AMSE (average mean square error), AMAE (average mean absolute error),
    and ARMSE (average root mean square error).
    """

    def __init__(self, window_size=200):
        super().__init__()
        self.n_targets = 0
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.total_square_error_correction = FastBuffer(window_size)
        self.average_error_correction = FastBuffer(window_size)
        self.window_size = window_size

    def reset(self):
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.total_square_error_correction = FastBuffer(self.window_size)
        self.average_error_correction = FastBuffer(self.window_size)

    def add_result(self, y, prediction):
        """ Use the true value and the prediction to update the statistics.

        Parameters
        ----------
        y: float or list or np.ndarray
            The true value(s).

        prediction: float or list or np.ndarray
            The predicted value(s).

        prediction: float or list or np.ndarray
            The predicted value(s).

        """

        self.last_true_label = y
        self.last_prediction = prediction

        m = 0
        if hasattr(y, 'size'):
            m = y.size
        elif hasattr(y, 'append'):
            m = len(y)
        self.n_targets = m

        self.total_square_error += (y - prediction) * (y - prediction)
        self.average_error += np.absolute(y - prediction)

        old_square = self.total_square_error_correction.add_element(
            np.array([-1 * (y - prediction) * (y - prediction)])
        )
        old_average = self.average_error_correction.add_element(
            np.array([-1 * (np.absolute(y - prediction))])
        )

        if (old_square is not None) and (old_average is not None):
            self.total_square_error += old_square[0]
            self.average_error += old_average[0]

    def get_average_mean_square_error(self):
        """ Computes the window/current average mean square error.

        Returns
        -------
        float
            The window/current average mean square error.
        """
        if self._sample_count == 0:
            return 0.0
        else:
            return np.sum(self.total_square_error / self._sample_count) \
                / self.n_targets

    def get_average_absolute_error(self):
        """ Computes the window/current average mean absolute error.

        Returns
        -------
        float
            The window/current average mean absolute error.
        """
        if self._sample_count == 0:
            return 0.0
        else:
            return np.sum(self.average_error / self._sample_count) \
                / self.n_targets

    def get_average_root_mean_square_error(self):
        """ Computes the mean square error.

        Returns
        -------
        float
            The average mean square error.
        """
        if self._sample_count == 0:
            return 0.0
        else:
            mse = self.total_square_error / self._sample_count
            return np.sum(
                np.sqrt(mse, out=np.zeros_like(mse), where=mse >= 0.0)
            ) / self.n_targets

    def get_last(self):
        return self.last_true_label, self.last_prediction

    @property
    def _sample_count(self):
        return self.total_square_error_correction.get_current_size()

    def get_info(self):
        return 'MultiTargetRegressionMeasurements: sample_count: ' + \
                str(self._sample_count) + ' - average_mean_square_error: ' + \
                str(self.get_average_mean_square_error()) + ' - average_mean_absolute_error: ' + \
                str(self.get_average_absolute_error()) + ' - average_root_mean_square_error: ' + \
                str(self.get_average_root_mean_square_error())


class RunningTimeMeasurements(object):
    """ Class used to compute the running time for each evaluated prediction
        model.

        The training, prediction, and total time are considered separately. The
        class accounts for the amount of time each model effectively spent on
        training and testing. To do so, timers for each of the actions are
        considered.

        Besides the properties getters, the available compute time methods
        must be used as follows:

        - `compute_{training, testing}_time_begin`
        - Perform training/action
        - `compute_{training, testing}_time_end`

        Additionally, the `update_time_measurements` method updates the total
        running time accounting, as well as, the total seen samples count.
    """

    def __init__(self):
        super().__init__()
        self._training_start = None
        self._testing_start = None
        self._training_time = 0
        self._testing_time = 0
        self._sample_count = 0
        self._total_time = 0

    def reset(self):
        self._training_time = 0
        self._testing_time = 0
        self._sample_count = 0
        self._total_time = 0

    def compute_training_time_begin(self):
        """ Initiates the training time accounting.
        """
        self._training_start = timer()

    def compute_training_time_end(self):
        """ Finishes the training time accounting. Updates current total
            training time.
        """
        self._training_time += timer() - self._training_start

    def compute_testing_time_begin(self):
        """ Initiates the testing time accounting.
        """
        self._testing_start = timer()

    def compute_testing_time_end(self):
        """ Finishes the testing time accounting. Updates current total
            testing time.
        """
        self._testing_time += timer() - self._testing_start

    def update_time_measurements(self, increment=1):
        """ Updates the current total running time. Updates the number of seen
        samples by `increment`.
        """
        if increment > 0:
            self._sample_count += increment
        else:
            self._sample_count += 1

        self._total_time = self._training_time + self._testing_time

    def get_current_training_time(self):
        return self._training_time

    def get_current_testing_time(self):
        return self._testing_time

    def get_current_total_running_time(self):
        return self._total_time

    def get_info(self):
        return 'RunningTimeMeasurements: sample_count: ' + \
                str(self._sample_count) + ' - Total running time: ' + \
                str(self.get_current_total_running_time()) + \
                ' - training_time: ' + \
                str(self.get_current_training_time()) + \
                ' - testing_time: ' + \
                str(self.get_current_testing_time())


def hamming_score(true_labels, predicts):
    """ Computes de hamming score, which is known as the label-based accuracy,
    designed for multi-label problems. It's defined as the number of correctly
    predicted labels divided by all classified labels.

    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true labels for all the classification tasks and for
        n_samples.

    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for
        n_samples.

    Returns
    -------
    float
        The hamming score, or label-based accuracy, for the given sets.

    Examples
    --------
    >>> from skmultiflow.metrics import hamming_score
    >>> true_labels = [[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,1,1]]
    >>> predictions = [[0,1,0,1],[0,1,1,0],[0,1,0,1],[1,1,1,1]]
    >>> hamming_score(true_labels, predictions)
    0.75

    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    return np.sum((true_labels == predicts) * 1.) / N / L


def j_index(true_labels, predicts):
    """ Computes the Jaccard Index of the given set, which is also called the
    'intersection over union' in multi-label settings. It's defined as the
    intersection between the true label's set and the prediction's set,
    divided by the sum, or union, of those two sets.

    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true labels for all the classification tasks and for
        n_samples.

    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for
        n_samples.

    Returns
    -------
    float
        The J-index, or 'intersection over union', for the given sets.

    Examples
    --------
    >>> from skmultiflow.metrics import j_index
    >>> true_labels = [[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,1,1]]
    >>> predictions = [[0,1,0,1],[0,1,1,0],[0,1,0,1],[1,1,1,1]]
    >>> j_index(true_labels, predictions)
    0.66666666666666663

    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    s = 0.0
    for i in range(N):
        inter = sum((true_labels[i, :] * predicts[i, :]) > 0) * 1.
        union = sum((true_labels[i, :] + predicts[i, :]) > 0) * 1.
        if union > 0:
            s += inter / union
        elif np.sum(true_labels[i, :]) == 0:
            s += 1.
    return s * 1. / N


def exact_match(true_labels, predicts):
    """ This is the most strict metric for the multi label setting. It's defined
    as the percentage of samples that have all their labels correctly classified.

    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true labels for all the classification tasks and for
        n_samples.

    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for
        n_samples.

    Returns
    -------
    float
        The exact match percentage between the given sets.

    Examples
    --------
    >>> from skmultiflow.metrics import exact_match
    >>> true_labels = [[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,1,1]]
    >>> predictions = [[0,1,0,1],[0,1,1,0],[0,1,0,1],[1,1,1,1]]
    >>> exact_match(true_labels, predictions)
    0.5

    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    return np.sum(np.sum((true_labels == predicts) * 1, axis=1) == L) * 1. / N
