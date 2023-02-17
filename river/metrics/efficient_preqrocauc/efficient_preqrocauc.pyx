# distutils: language = c++
import cython

from .efficient_preqrocauc cimport PreqROCAUC as CppPreqROCAUC

cdef class EfficientPreqROCAUC:
    cdef CppPreqROCAUC* preqrocauc

    def __cinit__(self, cython.int positiveLabel, cython.ulong windowSize):
        self.preqrocauc = new CppPreqROCAUC(positiveLabel, windowSize)

    def __dealloc__(self):
        del self.preqrocauc

    def update(self, label, score):
        self.preqrocauc.update(label, score)

    def revert(self, label, score):
        self.preqrocauc.revert(label, score)

    def get(self):
        return self.preqrocauc.get()
