cdef class Statistic:
    cpdef double get(self)

cdef class Univariate(Statistic):
    cpdef Univariate update(self, double x)

cdef class Bivariate(Statistic):
    cpdef Bivariate update(self, double x, double y)
