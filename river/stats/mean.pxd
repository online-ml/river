cimport river.stats.base

cdef class Mean(river.stats.base.Univariate):
    cdef readonly double mean
    cdef readonly double n

    cpdef double get(self)
    cpdef Mean update(self, double x, double w=*)
    cpdef Mean revert(self, double x, double w=*)
