import abc
import copy
import numbers
import typing

from river import base


class Literal(base.Base):
    __slots__ = "on, at"

    def __init__(self, on, at):
        self.on = on
        self.at = at

    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def describe(self):
        pass


class BinaryLiteral(Literal):
    def __init__(self, on, at):
        super().__init__(on, at)

    def __call__(self, x):
        if self.on in x:
            return x[self.on] <= self.at
        return False

    def describe(self):
        return f"{self.on} ≤ {self.at}"


class NominalLiteral(Literal):
    def __init__(self, on, at):
        super().__init__(on, at)

    def __call__(self, x):
        if self.on in x:
            return x[self.on] == self.at
        return False

    def describe(self):
        return f"{self.on} = {self.at}"


class Rule(base.Estimator, metaclass=abc.ABCMeta):
    def __init__(self, template_splitter, **attributes):
        self.template_splitter = template_splitter
        self.literals = []
        self.splitters = []

        self.__dict__.update(attributes)

    def expand(self):
        # TODO
        pass

    def covers(self, x):
        return all(map(lambda lit: lit(x), self.literals))

    @abc.abstractmethod
    def new_nominal_splitter(self):
        pass

    def _iter_features(self, x) -> typing.Iterable:
        """Determine how the input instance is looped through when updating the splitters.

        Parameters
        ----------
        x
            The input instance.
        """
        for att_id, att_val in x.items():
            yield att_id, att_val

    @abc.abstractmethod
    def _update_stats(self, y, w):
        pass

    def update(self, x, y, w):
        self._update_stats(y, w)

        for att_id, att_val in self._iter_features(x):
            try:
                splt = self.splitters[att_id]
            except KeyError:
                if isinstance(att_val, numbers.Number):
                    self.splitters[att_id] = copy.deepcopy(self.template_splitter)
                else:
                    self.splitters[att_id] = self.new_nominal_splitter()
                splt = self.splitters[att_id]
            splt.update(x, y, w)

    def __repr__(self):
        return f"{' and '.join([lit.describe() for lit in self.literals])}"
