# distutils: language = c++
# distutils: extra_compile_args = "-std=c++11"

import cython

from .efficient_rollingrocauc cimport RollingROCAUC as CppRollingROCAUC

cdef class EfficientRollingROCAUC:
    cdef cython.int positiveLabel
    cdef cython.ulong windowSize
    cdef CppRollingROCAUC* rollingrocauc

    def __cinit__(self, cython.int positiveLabel, cython.ulong windowSize):
        self.positiveLabel = positiveLabel
        self.windowSize = windowSize
        self.rollingrocauc = new CppRollingROCAUC(positiveLabel, windowSize)

    def __dealloc__(self):
        if not self.rollingrocauc == NULL:
            del self.rollingrocauc

    def update(self, label, score):
        self.rollingrocauc.update(label, score)

    def revert(self, label, score):
        self.rollingrocauc.revert(label, score)

    def get(self):
        return self.rollingrocauc.get()

    def __getnewargs_ex__(self):
        # Pickle will use this function to pass the arguments to __new__
        return (self.positiveLabel, self.windowSize),{}

    def __getstate__(self):
        """
            On pickling, the true labels and scores of the instances in the
            window will be dumped
        """
        return (self.rollingrocauc.getTrueLabels(), self.rollingrocauc.getScores())

    def __setstate__(self, state):
        """
            On unpickling, the state parameter will have the true labels
            and scores, this function updates the rollingrocauc with them
        """

        # Labels returned by __getstate__ are normalized (0 or 1)
        labels, scores = state

        for label, score in zip(labels, scores):
            # If label is 1, update with the positive label defined by the constructor
            # Else, update with a negative label
            l = self.positiveLabel if label else int(not self.positiveLabel)
            self.update(l, score)
