import abc


class SplitEnumerator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, values):
        """Yields splits."""


class UnaryEnumerator(SplitEnumerator):

    def __call__(self, values):
        """

        Example:

            >>> enum = UnaryEnumerator()
            >>> for left, right in enum(['a', 'b', 'c', 'd']):
            ...     print(left, right)
            ['a'] ['b', 'c', 'd']
            ['b'] ['a', 'c', 'd']
            ['c'] ['a', 'b', 'd']
            ['d'] ['a', 'b', 'c']

        """
        values = list(values)
        for i, v in enumerate(values):
            yield [v], values[:i] + values[i + 1:]


class ContiguousEnumerator(SplitEnumerator):

    def __call__(self, values):
        """

        Example:

            >>> enum = ContiguousEnumerator()
            >>> for left, right in enum(['a', 'b', 'c', 'd']):
            ...     print(left, right)
            ['a'] ['b', 'c', 'd']
            ['a', 'b'] ['c', 'd']
            ['a', 'b', 'c'] ['d']

        """
        values = list(values)
        for i in range(1, len(values)):
            yield values[:i], values[i:]
