import abc


cdef class Statistic:
    """A statistic."""

    # Define the format specification used for string representation.
    _fmt = ',.6f'  # Use commas to separate big numbers and show 6 decimals

    cpdef double get(self):
        """Returns the current value of the statistic."""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.get():{self._fmt}}'.rstrip('0')


cdef class Univariate(Statistic):
    """A univariate statistic measures a property of a variable."""

    cpdef Univariate update(self, double x):
        """Updates and returns the called instance."""
        raise NotImplementedError

    cpdef Univariate revert(self, double x):
        """Reverts and returns the called instance."""
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def __or__(self, other):
        from .link import Link
        return Link(left=self, right=other)


class RollingUnivariate(Univariate):
    """A rolling univariate statistic measures a property of a variable over a window."""

    @abc.abstractproperty
    def window_size(self):
        pass

    @property
    def name(self):
        return f'{super().name}_{self.window_size}'


cdef class Bivariate(Statistic):
    """A bivariate statistic measures a relationship between two variables."""

    cpdef Bivariate update(self, double x, double y):
        """Updates and returns the called instance."""
        raise NotImplementedError
