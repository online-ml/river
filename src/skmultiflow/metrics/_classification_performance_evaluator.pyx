cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import warnings
from cpython cimport bool
from collections import deque

from ._confusion_matrix import ConfusionMatrix, MultiLabelConfusionMatrix

DTYPE = np.float
ctypedef np.float_t DTYPE_t


cdef class ClassificationPerformanceEvaluator:
    """ Classification performance evaluator.

    Tracks a classifier's performance and provides, at any moment, updated
    performance metrics. This performance evaluator is designed for single-output
    (binary and multi-class) classification tasks.

    Parameters
    ----------
    n_classes: int, optional (default=2)
        The number of classes.

    Notes
    -----
    Although the number of classes can be defined (default=2 for the binary case),
    if more classes are observed, then the confusion matrix is reshaped to account
    for new (emerging) classes.

    """

    def __init__(self, int n_classes=2):
        if n_classes < 2:
            warnings.warn('n_classes can not be less than 2, using default (2)')
            n_classes = 2
        self._init_n_classes = n_classes
        self.confusion_matrix = ConfusionMatrix(n_classes=n_classes)
        self.n_samples = 0
        self.last_true = 0
        self.last_pred = 0
        self.weight_majority_classifier = 0.0
        self.weight_correct_no_change_classifier = 0.0
        self._total_weight_observed = 0.0

    cpdef void add_result(self, int y_true, int y_pred, double sample_weight=1.0):
        """ Updates internal statistics with the results of a prediction.

        Parameters
        ----------
        y_true: int
            The true (actual) value.

        y_pred: int
            The predicted value.

        sample_weight: float
            The weight of the sample.

        """
        self.n_samples += 1

        if sample_weight > 0.0:
            self.confusion_matrix[(y_true, y_pred)] = sample_weight
            self._total_weight_observed += sample_weight

        if self.majority_class() == y_true:
            self.weight_majority_classifier += sample_weight

        if self.last_true == y_true:
            self.weight_correct_no_change_classifier += sample_weight

        # Keep track of last entry
        self.last_true = y_true
        self.last_pred = y_pred

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef int majority_class(self):
        """ Computes the majority class.

        Returns
        -------
        int
            The majority class.

        """
        if self._total_weight_observed == 0:
            return 0
        cdef int majority_idx = 0
        cdef double max_prob = 0.0
        cdef double sum_value
        for i in sorted(self.confusion_matrix.classes):
            sum_value = self.confusion_matrix.sum_row[i] / self.n_samples
            if sum_value > max_prob:
                max_prob = sum_value
                majority_idx = i
        return majority_idx

    cpdef double accuracy_score(self):
        """ Accuracy score.
        
        The accuracy is the ratio or correctly classified samples to the total
        number of samples.

        Returns
        -------
        float
            Accuracy.

        """
        return (self.confusion_matrix.sum_diag / self.n_samples)\
            if self.n_samples > 0 else 0.0

    @property
    def total_weight_observed(self):
        """ The total weight observed.
        """
        return self._total_weight_observed

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef double kappa_score(self):
        """ Kappa score.
         
        Cohen's Kappa [1]_ expresses the level of agreement between two annotators
         on a classification problem. It is defined as
         
        .. math::
            \kappa = (p_o - p_e) / (1 - p_e)
        
        where :math:`p_o` is the empirical probability of agreement on the label
        assigned to any sample (prequential accuracy), and :math:`p_e` is
        the expected agreement when both annotators assign labels randomly.

        Returns
        -------
        float
            Cohen's Kappa.
            
        References
        ----------
        .. [1] J. Cohen (1960). "A coefficient of agreement for nominal scales".
               Educational and Psychological Measurement 20(1):37-46.
               doi:10.1177/001316446002000104.

        """
        cdef double p0 = self.accuracy_score()
        cdef double pe = 0.0
        cdef double estimation_row
        cdef double estimation_col

        if self._total_weight_observed > 0.0:
            for i in self.confusion_matrix.classes:
                estimation_row = self.confusion_matrix.sum_row[i] / self.n_samples
                estimation_col = self.confusion_matrix.sum_col[i] / self.n_samples
                pe += estimation_row * estimation_col
            # if ill-defined, return 0.0
            return (p0 - pe) / (1.0 - pe) if pe != 1.0 else 0.0
        return 0.0

    cpdef double kappa_t_score(self):
        """ Kappa-T score.
        
        The Kappa Temp [1]_ measures the temporal correlation between samples.
        It is defined as
         
        .. math::
            \kappa_{t} = (p_o - p_e) / (1 - p_e)
        
        where :math:`p_o` is the empirical probability of agreement on the label
        assigned to any sample (prequential accuracy), and :math:`p_e` is
        the prequential accuracy of the ``no-change classifier`` that predicts
        only using the last class seen by the classifier.

        Returns
        -------
        float
            Kappa-T.
        
        References
        ----------
        .. [1] A. Bifet et al. (2013). "Pitfalls in benchmarking data stream classification
               and how to avoid them." Proc. of the European Conference on Machine Learning
               and Principles and Practice of Knowledge Discovery in Databases (ECMLPKDD'13),
               Springer LNAI 8188, p. 465-479.

        """
        cdef double p0 = self.accuracy_score()
        cdef double pe = 0.0
        if self._total_weight_observed > 0.0:
            if self.n_samples > 0:
                pe = self.weight_correct_no_change_classifier / self.n_samples
            # if ill-defined, return 0.0
            return (p0 - pe) / (1.0 - pe) if pe != 1.0 else 0.0
        return 0.0

    cpdef double kappa_m_score(self):
        """ Kappa-M score.
        
        The Kappa-M statistic [1]_ compares performance with the majority class classifier.
        It is defined as
         
        .. math::
            \kappa_{m} = (p_o - p_e) / (1 - p_e)
        
        where :math:`p_o` is the empirical probability of agreement on the label
        assigned to any sample (prequential accuracy), and :math:`p_e` is
        the prequential accuracy of the ``majority classifier``.

        Returns
        -------
        float
            Kappa-M.
        
        References
        ----------
        .. [1] A. Bifet et al. "Efficient online evaluation of big data stream classifiers."
               In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery
               and data mining, pp. 59-68. ACM, 2015.

        """
        cdef double p0 = self.accuracy_score()
        cdef double pe = 0.0
        if self._total_weight_observed > 0.0:
            if self.n_samples > 0:
                pe = self.weight_majority_classifier / self.n_samples
            # if ill-defined, return 0.0
            return (p0 - pe) / (1.0 - pe) if pe != 1.0 else 0.0
        return 0.0

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef double precision_score(self, int class_value=-1):
        """ Precision score.
        
        The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
        true positives and ``fp`` the number of false positives.
        
        Parameters
        ----------
        class_value: int, optional (default=-1)
            Class value to calculate this metric for. Not used by default.

        Returns
        -------
        float
            Precision.
        
        Notes
        -----
        If seen data corresponds to a multi-class problem then calculates the ``macro``
        average, this is, calculates metrics for each class, and find their unweighted mean.

        """
        cdef double total
        cdef int n_classes = self.confusion_matrix.n_classes
        if n_classes == 2 and class_value == -1:
            # Binary case
            return self._precision_per_class(1)
        elif class_value == -1:
            # Multi-class case, calculate 'macro' average
            total = 0.0
            for i in self.confusion_matrix.classes:
                total += self._precision_per_class(i)
            return total / <double>n_classes
        else:
            # Calculate for specified class
            return self._precision_per_class(class_value)

    cdef double _precision_per_class(self, int class_value):
        cdef double tp = self.confusion_matrix[(class_value, class_value)]
        cdef double tp_plus_fp = 0.0
        if self.confusion_matrix.n_classes > 0 and class_value >= 0:
            tp_plus_fp = self.confusion_matrix.sum_col[class_value]
            # if ill-defined, return 0.0
            return tp / tp_plus_fp if tp_plus_fp > 0.0 else 0.0
        return 0.0

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef double recall_score(self, int class_value=-1):
        """ Recall score.
        
        The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
        true positives and ``fn`` the number of false negatives.
        
        Parameters
        ----------
        class_value: int, optional (default=-1)
            Class value to calculate this metric for. Not used by default.
        
        Returns
        -------
        float
            Recall.
        
        Notes
        -----
        If seen data corresponds to a multi-class problem then calculates the ``macro``
        average, this is, calculates metrics for each class, and find their unweighted mean.
        
        """
        cdef double total
        cdef int n_classes = self.confusion_matrix.n_classes
        if n_classes == 2 and class_value == -1:
            # Binary case
            return self._recall_per_class(1)
        elif class_value == -1:
            # Multi-class case, calculate 'macro' average
            total = 0.0
            for i in self.confusion_matrix.classes:
                total += self._recall_per_class(i)
            return total / <double>n_classes
        else:
            # Calculate for specified class
            return self._recall_per_class(class_value)

    cdef double _recall_per_class(self, int class_value):
        cdef double tp = self.confusion_matrix[(class_value, class_value)]
        cdef double tp_plus_fn = 0.0
        if self.confusion_matrix.n_classes > 0 and class_value >= 0:
            tp_plus_fn = self.confusion_matrix.sum_row[class_value]
            # if ill-defined, return 0.0
            return tp / tp_plus_fn if tp_plus_fn > 0.0 else 0.0
        return 0.0

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef double f1_score(self, int class_value=-1):
        r""" F1 score.
        
        The F1 score can be interpreted as a weighted average of the precision and
        recall. The relative contribution of precision and recall to the F1 score
        are equal. The F1 score is defined as:
        
        .. math::
            F1 = \frac{2 \times (precision \times recall)}{(precision + recall)}
            
        Parameters
        ----------
        class_value: int, optional (default=-1)
            Class value to calculate this metric for. Not used by default.

        Returns
        -------
        float
            F1-score.
        
        Notes
        -----
        If seen data corresponds to a multi-class problem then calculates the ``macro``
        average, this is, calculates metrics for each class, and find their unweighted mean.

        """
        cdef double total
        cdef int n_classes = self.confusion_matrix.n_classes
        cdef double sum_f1
        if n_classes == 2:
            # Binary case
            return self._f1_per_class(1)
        if class_value == -1 and class_value == -1:
            # Multi-class case, calculate 'macro' average
            sum_f1 = 0.0
            for i in self.confusion_matrix.classes:
                sum_f1 += self._f1_per_class(i)
            return sum_f1 / <double>n_classes
        else:
            # Calculate for specified class
            return self._f1_per_class(class_value)

    cdef _f1_per_class(self, int class_value):
        cdef double precision
        cdef double recall
        if self.confusion_matrix.n_classes > 0:
            precision = self.precision_score(class_value)
            recall = self.recall_score(class_value)
            # if ill-defined, return 0.0
            return (2.0 * (precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        return 0.0

    cpdef double geometric_mean_score(self):
        """ Geometric mean score.
        
        The geometric mean is a good indicator of a classifier's performance
        in the presence of class imbalance because it is independent of the
        distribution of examples between classes [1]_. This implementation
        computes the geometric mean of class-wise sensitivity (recall)
        
        .. math::
            gm = \sqrt[n]{s_1\cdot s_2\cdot s_3\cdot \ldots\cdot s_n}
        
        where :math:`s_i` is the sensitivity (recall) of class :math:`i` and : math: `n`
        is the number of classes.
        
        Returns
        -------
        float
            Geometric mean score.
        
        References
        ----------
        .. [1] Barandela, R. et al. “Strategies for learning in class imbalance problems”,
               Pattern Recognition, 36(3), (2003), pp 849-851.
        
        """
        cdef int n_classes = self.confusion_matrix.n_classes
        cdef np.ndarray[DTYPE_t, ndim=1] sensitivity_per_class = np.zeros(n_classes, dtype=DTYPE)
        cdef unsigned int i, c
        if n_classes > 0:
            for i, c in enumerate(self.confusion_matrix.classes):
                sensitivity_per_class[i] = self._recall_per_class(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                return <double>sp.stats.gmean(sensitivity_per_class)
        return 0.0

    # specificity was used for the calculation of geometric mean for binary classification.
    # It is not required in the new implementation. However, it might be handy in the future.
    # cdef double _specificity_per_class(self, int class_value):
    #     cdef double tn
    #     cdef double fp
    #     if self.confusion_matrix.n_classes > 0:
    #         tn = self.confusion_matrix.sum_diag - self.confusion_matrix[(class_value, class_value)]
    #         fp = self.confusion_matrix.sum_col[class_value] - self.confusion_matrix[(class_value, class_value)]
    #         # if ill-defined, return 0.0
    #         return tn / (tn + fp) if (tn + fp) > 0.0 else 0.0
    #     return 0.0

    def get_last(self):
        """ Last samples (y_true, y_pred) observed.

        Returns
        -------
        tuple
            (last_true, last_pred) tuple

        """
        return self.last_true, self.last_pred

    def reset(self):
        """ Reset the evaluator to its initial state.
        """
        self.__init__(n_classes=self._init_n_classes)

    def get_info(self):
        """ Get (current) information about the performance evaluator.
        """
        return '{}('.format(type(self).__name__) + \
               'n_classes={}, '.format(self.confusion_matrix.n_classes) + \
               'n_samples={}, '.format(self.n_samples) + \
               self._get_info_metrics() + ')'

    def _get_info_metrics(self):
        return 'accuracy_score={:.6f}, '.format(self.accuracy_score()) + \
               'kappa_m_score={:.6f}, '.format(self.kappa_m_score()) + \
               'kappa_t_score={:.6f}, '.format(self.kappa_t_score()) + \
               'kappa_m_score={:.6f}, '.format(self.kappa_m_score()) + \
               'precision_score={:.6f}, '.format(self.precision_score()) + \
               'recall_score={:.6f}, '.format(self.recall_score()) + \
               'f1_score={:.6f}, '.format(self.f1_score()) + \
               'geometric_mean_score={:.6f}, '.format(self.geometric_mean_score()) + \
               'majority_class={}'.format(self.majority_class())


cdef class WindowClassificationPerformanceEvaluator(ClassificationPerformanceEvaluator):
    """ Window classification performance evaluator.

    Tracks a classifier's performance over a sliding window and provides, at any moment,
    updated performance metrics. This performance evaluator is designed for single-output
    (binary and multi-class) classification tasks.

    Parameters
    ----------
    n_classes: int, optional (default=2)
        The number of classes.

    window_size: int, optional (default=200)
        The size of the window.

    Notes
    -----
    Although the number of classes can be defined (default=2 for the binary case),
    if more classes are observed, then the confusion matrix is reshaped to account
    for new (emerging) classes.

    """

    def __init__(self, int n_classes=2, int window_size=200):
        super().__init__(n_classes=n_classes)
        self.window_size = window_size
        self._queue = deque()

    cpdef void add_result(self, int y_true, int y_pred, double sample_weight=1.0):
        """ Updates internal statistics with the results of a prediction.

        Parameters
        ----------
        y_true: int
            The true (actual) value.

        y_pred: int
            The predicted value.

        sample_weight: float
            The weight of the sample.
        
        Notes
        -----
        Oldest samples are automatically removed when the window is full. Special care
        is taken to keep internal statistics consistent with the samples in the window.

        """
        cdef (int, int, double, int, int) remove_tuple
        cdef int y_true_remove = 0
        cdef int y_pred_remove = 0
        cdef double weight_remove = 0.0
        cdef int majority_classifier_correction = 0
        cdef int correct_no_change_classifier_correction = 0

        if len(self._queue) == self.window_size:
            # The window is full, remove older samples before adding new samples
            remove_tuple = self._queue.popleft()
            y_true_remove = remove_tuple[0]
            y_pred_remove = remove_tuple[1]
            weight_remove = remove_tuple[2]
            majority_classifier_correction = remove_tuple[3]
            correct_no_change_classifier_correction = remove_tuple[4]
            # Update confusion matrix by removing the oldest samples
            self.confusion_matrix[(y_true_remove, y_pred_remove)] = -weight_remove
            # Update the majority and correct-no-change classifiers' statistics
            self.weight_majority_classifier -= weight_remove if majority_classifier_correction == 1 else 0.0
            self.weight_correct_no_change_classifier -= \
                weight_remove if correct_no_change_classifier_correction == 1 else 0.0
            # Update sample count
            self.n_samples -= 1

        self.n_samples += 1

        if sample_weight > 0.0:
            self.confusion_matrix[(y_true, y_pred)] = sample_weight
            self._total_weight_observed += sample_weight

        majority_classifier_correction = 0
        if self.majority_class() == y_true:
            self.weight_majority_classifier += sample_weight
            majority_classifier_correction = 1

        correct_no_change_classifier_correction = 0
        if self.last_true == y_true:
            self.weight_correct_no_change_classifier += sample_weight
            correct_no_change_classifier_correction = 1

        # Keep track of last entry
        self.last_true = y_true
        self.last_pred = y_pred

        # Enqueue sample so it can be removed when the window is full.
        self._queue.append((y_true,
                            y_pred,
                            sample_weight,
                            majority_classifier_correction,
                            correct_no_change_classifier_correction))

    def reset(self):
        """ Reset the evaluator to its initial state.
        """
        self.__init__(window_size=self.window_size)

    def get_info(self):
        """ Get (current) information about the performance evaluator.
        """
        return '{}('.format(type(self).__name__) + \
               'n_classes={}, '.format(self.confusion_matrix.n_classes) + \
               'window_size={}, '.format(self.window_size) + \
               'n_samples={}, '.format(self.n_samples) + \
               self._get_info_metrics() + ')'


cdef class MultiLabelClassificationPerformanceEvaluator:
    """ Multi-label classification performance evaluator.

    Tracks a classifier's performance and provides, at any moment, updated
    performance metrics. This performance evaluator is designed for multi-output
    (multi-label) classification tasks.

    Parameters
    ----------
    n_labels: int, optional (default=2)
        The number of labels.

    Notes
    -----
    Although the number of labels can be defined (default=2), if more labels are observed,
    then the confusion matrix is reshaped to account for new (emerging) labels.

    """

    def __init__(self, int n_labels=2):
        if n_labels < 2:
            warnings.warn('n_labels can not be less than 2, using default (2)')
            n_labels = 2
        self._init_n_labels = n_labels
        self.confusion_matrix = MultiLabelConfusionMatrix()
        self.last_true = np.zeros(n_labels, dtype=DTYPE)
        self.last_pred = np.zeros(n_labels, dtype=DTYPE)
        self.n_samples = 0
        self.exact_match_cnt = 0
        self.jaccard_sum = 0.0

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef void add_result(self, np.ndarray y_true, np.ndarray y_pred, double sample_weight=1.0):
        """ Updates internal statistics with the results of a prediction.

        Parameters
        ----------
        y_true: np.ndarray of shape (n_labels,)
            A 1D array with binary indicators for true (actual) values.

        y_pred: np.ndarray of shape (n_labels,)
            A 1D array with binary indicators for predicted values.

        sample_weight: float
            The weight of the sample.

        """
        cdef unsigned int is_equal = 1
        cdef double inter_val
        cdef double union_val

        if not _check_multi_label_inputs(y_true, y_pred):
            return

        self.n_samples += 1

        for i in range(len(y_true)):
            self.confusion_matrix[(i, y_true[i], y_pred[i])] = sample_weight
            if y_true[i] != y_pred[i]:
                is_equal = 0

        # Update exact match
        self.exact_match_cnt += is_equal

        # Update jaccard index sum
        inter_val = np.sum(np.logical_and(y_true, y_pred))
        union_val = np.sum(np.logical_or(y_true, y_pred))
        # skip if ill-defined
        if union_val > 0:
            self.jaccard_sum += inter_val / union_val

        # Keep track of last entry
        self.last_true = y_true
        self.last_pred = y_pred

    cpdef double hamming_score(self):
        """ Hamming score.
        
        The Hamming score is the fraction of labels that are correctly predicted.
        
        Returns
        -------
        float
            Hamming score.

        """
        try:
            return np.sum(self.confusion_matrix.data[:,1,1]) / \
                   (self.n_samples * self.confusion_matrix.n_labels)
        except ZeroDivisionError:
            return 0.0   # ill-defined, return 0.0

    cpdef double hamming_loss_score(self):
        """ Hamming loss score.
        
        The Hamming loss is the complement of the Hamming score.
        
        Returns
        -------
        float
            Hamming loss score.

        """
        return 1.0 - self.hamming_score()

    cpdef double exact_match_score(self):
        """ Exact match score.
        
        This is the most strict multi-label metric, defined as the number of
        samples that have all their labels correctly classified, divided by the
        total number of samples.
        
        Returns
        -------
        float
            Exact match score.

        """
        try:
            return self.exact_match_cnt / self.n_samples
        except ZeroDivisionError:
            return 0.0   # ill-defined, return 0.0

    cpdef double jaccard_score(self):
        """ Jaccard similarity coefficient score.
        
        The Jaccard index, or Jaccard similarity coefficient, defined as
        the size of the intersection divided by the size of the union of two label
        sets, is used to compare set of predicted labels for a sample to the
        corresponding set of labels in ``y_true``.
        
        Returns
        -------
        float
            Jaccard score.
        
        Notes
        -----
        The Jaccard index may be a poor metric if there are no positives for some samples or labels.
        The Jaccard index is undefined if there are no true or predicted labels, this implementation
        will return a score of 0 if this is the case.
        
        """
        try:
            return self.jaccard_sum / self.n_samples
        except ZeroDivisionError:
            return 0.0   # ill-defined, return 0.0

    def get_last(self):
        """ Last samples (y_true, y_pred) observed.

        Returns
        -------
        tuple
            (last_true, last_pred) tuple

        """
        return self.last_true, self.last_pred

    def reset(self):
        """ Reset the evaluator to its initial state.
        """
        self.__init__()

    def get_info(self):
        """ Get (current) information about the performance evaluator.
        """
        return '{}('.format(type(self).__name__) + \
               'n_labels={}, '.format(self.confusion_matrix.n_labels) + \
               'n_samples={}, '.format(self.n_samples) + \
               self._get_info_metrics() + ')'

    def _get_info_metrics(self):
        return 'hamming_score={:.6f}, '.format(self.hamming_score()) + \
               'hamming_loss_score={:.6f}, '.format(self.hamming_loss_score()) + \
               'exact_match_score={:.6f}, '.format(self.exact_match_score()) + \
               'jaccard_score={:.6f}'.format(self.jaccard_score())


cdef class WindowMultiLabelClassificationPerformanceEvaluator(MultiLabelClassificationPerformanceEvaluator):
    """ Window multi-label classification performance evaluator.

    Tracks a classifier's performance over a sliding window and provides, at any moment,
    updated performance metrics. This performance evaluator is designed for multi-output
    (multi-label) classification tasks.

    Parameters
    ----------
    n_labels: int, optional (default=2)
        The number of labels.

    window_size: int, optional (default=200)
        The size of the window.

    Notes
    -----
    Although the number of labels can be defined (default=2), if more labels are observed,
    then the confusion matrix is reshaped to account for new (emerging) labels.

    """


    def __init__(self, int n_labels=2, int window_size=200):
        super().__init__(n_labels=n_labels)
        self.window_size = window_size
        self._queue = deque()

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef void add_result(self, np.ndarray y_true, np.ndarray y_pred, double sample_weight=1.0):
        """ Updates internal statistics with the results of a prediction.

        Parameters
        ----------
        y_true: np.ndarray of shape (n_labels,)
            A 1D array with binary indicators for true (actual) values.

        y_pred: np.ndarray of shape (n_labels,)
            A 1D array with binary indicators for predicted values.

        sample_weight: float
            The weight of the sample.
        
        Notes
        -----
        Oldest samples are automatically removed when the window is full. Special care
        is taken to keep internal statistics consistent with the samples in the window.


        """
        cdef remove_tuple
        cdef y_true_remove
        cdef y_pred_remove
        cdef double weight_remove = 0.0
        cdef int is_equal_remove = 0
        cdef double jaccard_sum_remove = 0.0
        cdef unsigned int is_equal = 1
        cdef double inter_val
        cdef double union_val

        if not _check_multi_label_inputs(y_true, y_pred):
            return

        if len(self._queue) == self.window_size:
            # The window is full, remove older samples before adding new samples
            remove_tuple = self._queue.popleft()
            y_true_remove = remove_tuple[0]
            y_pred_remove = remove_tuple[1]
            weight_remove = remove_tuple[2]
            is_equal_remove = remove_tuple[3]
            jaccard_sum_remove = remove_tuple[4]
            # Update confusion matrix by removing the oldest samples
            for i in range(len(y_true_remove)):
                self.confusion_matrix[(i, y_true_remove[i], y_pred_remove[i])] = -weight_remove
            # Update the exact_no_match count
            self.exact_match_cnt -= is_equal_remove
            # Update the jaccard sum
            self.jaccard_sum -= jaccard_sum_remove
            # Update sample count
            self.n_samples -= 1

        for i in range(len(y_true)):
            self.confusion_matrix[(i, y_true[i], y_pred[i])] = sample_weight
            if y_true[i] != y_pred[i]:
                is_equal = 0

        self.n_samples += 1

        # Update exact match
        self.exact_match_cnt += is_equal
        is_equal_remove = is_equal

        # Update jaccard index sum
        jaccard_sum_remove = 0.0
        inter_val = np.sum(np.logical_and(y_true, y_pred))
        union_val = np.sum(np.logical_or(y_true, y_pred))
        # skip if ill-defined
        if union_val > 0:
            self.jaccard_sum += inter_val / union_val
            jaccard_sum_remove = inter_val / union_val

        # Keep track of last entry
        self.last_true = y_true
        self.last_pred = y_pred

        # Enqueue sample so it can be removed when the window is full.
        self._queue.append((y_true,
                            y_pred,
                            sample_weight,
                            is_equal_remove,
                            jaccard_sum_remove))

    def reset(self):
        """ Reset the evaluator to its initial state.
        """
        self.__init__(window_size=self.window_size)

    def get_info(self):
        """ Get (current) information about the performance evaluator.
        """
        return '{}('.format(type(self).__name__) + \
               'n_labels={}, '.format(self.confusion_matrix.n_labels) + \
               'window_size={}, '.format(self.window_size) + \
               'n_samples={}, '.format(self.n_samples) + \
               self._get_info_metrics() + ')'


cdef bool _check_multi_label_inputs(np.ndarray y_true, np.ndarray y_pred):
    """ Checks multi-label inputs
    
    Parameters
    ----------
    y_true: np.ndarray of shape (n_labels,)
        True (actual) values array
    y_pred: np.ndarray of shape (n_labels,)
        Predicted values array
    Returns
    -------
    bool
        True if valid, False otherwise.
    
    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths

    """

    cdef bool is_valid = True

    if len(y_true) != len(y_pred):
        raise ValueError('Inputs must have the same length, received: y_true {}, y_pred {}'.format(len(y_true),
                                                                                                   len(y_pred)))

    if np.any((np.isnan(y_true), np.isnan(y_pred))):
        warnings.warn("NaN values found, skipping sample.")
        is_valid = False

    return is_valid