# See _confusion_matrix.pyx for implementation details.

cdef class ConfusionMatrix:
    # This confusion matrix is a 2D matrix of shape ``(n_classes, n_classes)``, corresponding
    # to a single-target (binary and multi-class) classification task.

    # Internal variables
    cdef readonly set _init_classes             # Initial set of classes
    cdef readonly double sum_diag               # Sum across the diagonal
    cdef readonly sum_row                       # Sum per row
    cdef readonly sum_col                       # Sum per column
    cdef readonly data                          # The actual data (dictionary)
    cdef readonly int n_samples                 # Number of samples seen
    cdef readonly float total_weight            # Sum of sample_weights seen
    cdef readonly last_y_true                   # Last y_true value seen
    cdef readonly last_y_pred                   # Last y_pred value seen
    cdef readonly sample_correction             # Used to apply corrections during revert
    cdef readonly float weight_majority_classifier      # Correctly classified: majority class
    cdef readonly float weight_no_change_classifier     # Correctly classified: no-change

    # Methods
    cdef _majority_class(self)

cdef class MultiLabelConfusionMatrix:
    # This confusion matrix corresponds to a 3D matrix of shape ``(n_labels, 2, 2)`` meaning
    # that each ``label`` has a corresponding binary ``(2x2)`` confusion matrix.

    # Internal variables
    cdef readonly set _init_labels              # Initial set of labels
    cdef readonly set labels                    # Set of labels
    cdef readonly int n_labels                  # Number of labels
    cdef readonly data                          # The actual data (3D np.ndarray)
    cdef readonly dict _label_dict              # Dictionary to map labels and their label-index
    cdef readonly int _label_idx_cnt            # Internal label-index counter
    cdef readonly last_y_true                   # Last y_true value seen
    cdef readonly last_y_pred                   # Last y_pred value seen
    cdef readonly int n_samples                 # Number of samples seen
    cdef readonly sample_correction             # Used to apply corrections during revert
    cdef readonly int exact_match_cnt           # Exact match count
    cdef readonly double precision_sum          # Precision sum
    cdef readonly double recall_sum             # Recall sum
    cdef readonly double jaccard_sum            # Jaccard-index sum

    # Methods
    cdef int _map_label(self, label, bint add_label)
    cdef void _add_label(self, label)
    cdef void _reshape(self)
