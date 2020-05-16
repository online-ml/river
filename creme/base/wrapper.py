import abc

from creme import base


class Wrapper(base.Estimator):
    """A wrapper model."""

    @abc.abstractproperty
    def _wrapped_model(self):
        """Provides access to the wrapped model."""

    @property
    def _labelloc(self):
        """Indicates location of the wrapper name when drawing pipelines."""
        return 't'  # for top

    def __str__(self):
        return f'{type(self).__name__}({self._wrapped_model})'
