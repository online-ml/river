cimport base

cdef class Mean(base.Univariate):
    cdef readonly long n
    cdef readonly double mean
