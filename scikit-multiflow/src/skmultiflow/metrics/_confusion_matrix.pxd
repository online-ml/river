# See _confusion_matrix.pyx for implementation details.

cdef class ConfusionMatrix:
    # This confusion matrix is a 2D matrix of shape ``(n_classes, n_classes)``, corresponding
    # to a single-target (binary and multi-class) classification task.

    # Internal variables
    cdef readonly int _init_n_classes           # Initial number of classes
    cdef readonly set classes                   # Set of class-labels
    cdef readonly int n_classes                 # Number of classes
    cdef readonly double sum_diag               # Sum across the diagonal
    cdef readonly sum_row                       # Sum per row
    cdef readonly sum_col                       # Sum per column
    cdef readonly data                          # The actual data (dictionary)

cdef class MultiLabelConfusionMatrix:
    # This confusion matrix corresponds to a 3D matrix of shape ``(n_labels, 2, 2)`` meaning
    # that each ``label`` has a corresponding binary ``(2x2)`` confusion matrix.

    # Internal variables
    cdef readonly int _init_n_labels            # Initial number of labels
    cdef readonly set labels                    # Set of labels
    cdef readonly int n_labels                  # Number of labels
    cdef readonly int _max_label                # Max label observed
    cdef readonly data                          # The actual data (3D np.ndarray)

    # Methods
    cdef void _reshape(self)
