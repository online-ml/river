import numpy as np
cimport numpy as np
from cpython cimport bool

# See _classification_performance_evaluator.pyx for implementation details.

cdef class ClassificationPerformanceEvaluator:
    # Tracks a classifier's performance and provides, at any moment, updated
    # performance metrics. This performance evaluator is designed for single-output
    # (binary and multi-class) classification tasks.

    # Internal variables
    cdef readonly confusion_matrix                              # Confusion matrix
    cdef readonly int n_samples                                 # Number of samples observed
    cdef readonly int last_true                                 # Last true value observed
    cdef readonly int last_pred                                 # Last predicted value observed
    cdef readonly double weight_majority_classifier             # Weight seen by the majority class classifier
    cdef readonly double weight_correct_no_change_classifier    # Weight seen by the no-change classifier
    cdef readonly double _total_weight_observed                 # Total weight observed
    cdef readonly int _init_n_classes                           # Initial number of classes

    # Methods
    cpdef void add_result(self, int y_true, int y_pred, double sample_weight=*)
    cpdef int majority_class(self)
    cpdef double accuracy_score(self)
    cpdef double kappa_score(self)
    cpdef double kappa_t_score(self)
    cpdef double kappa_m_score(self)
    cpdef double precision_score(self, int class_value=*)
    cdef double _precision_per_class(self, int class_value)
    cpdef double recall_score(self, int class_value=*)
    cdef double _recall_per_class(self, int class_value)
    cpdef double f1_score(self, int class_value=*)
    cdef _f1_per_class(self, int class_value)
    cpdef double geometric_mean_score(self)
    # cdef double _specificity_per_class(self, int class_value)

cdef class WindowClassificationPerformanceEvaluator(ClassificationPerformanceEvaluator):
    # Tracks a classifier's performance over a sliding window and provides, at any moment,
    # updated performance metrics. This performance evaluator is designed for single-output
    # (binary and multi-class) classification tasks.

    # Internal variables
    cdef readonly _queue            # Queue to track samples in the window
    cdef int window_size            # Size of the window

cdef class MultiLabelClassificationPerformanceEvaluator():
    # Tracks a classifier's performance and provides, at any moment, updated
    # performance metrics. This performance evaluator is designed for multi-output
    # (multi-label) classification tasks.

    # Internal variables
    cdef readonly confusion_matrix          # Confusion matrix
    cdef readonly int n_samples             # Number of samples observed
    cdef readonly last_true                 # Last true value observed
    cdef readonly last_pred                 # Last predicted value observed
    cdef readonly int exact_match_cnt       # Exact match count
    cdef readonly double jaccard_sum        # Jaccard-index sum
    cdef readonly int _init_n_labels        # Initial number of labels

    # Methods
    cpdef void add_result(self, np.ndarray y_true, np.ndarray y_pred, double sample_weight=*)
    cpdef double hamming_score(self)
    cpdef double hamming_loss_score(self)
    cpdef double exact_match_score(self)
    cpdef double jaccard_score(self)

cdef class WindowMultiLabelClassificationPerformanceEvaluator(MultiLabelClassificationPerformanceEvaluator):
    # Tracks a classifier's performance over a sliding window and provides, at any moment,
    # updated performance metrics. This performance evaluator is designed for multi-output
    # (multi-label) classification tasks.

    # Internal variables
    cdef readonly _queue            # Queue to track samples in the window
    cdef int window_size            # Size of the window

cdef bool _check_multi_label_inputs(np.ndarray y_true, np.ndarray y_pred)
    # Utility function to check multi-label inputs