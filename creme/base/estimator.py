import abc
import copy
import inspect
import sys

from .. import utils


DEFAULT_TAGS = {
    'handles_text': False,
    'requires_positive_data': False,
    'handles_categorical_features': False
}


def _update_if_consistent(dict1, dict2):
    common_keys = set(dict1.keys()).intersection(dict2.keys())
    for key in common_keys:
        if dict1[key] != dict2[key]:
            raise TypeError(
                f'Inconsistent values for tag {key}: {dict1[key]} != {dict2[key]}')
    dict1.update(dict2)


class Estimator(abc.ABC):
    """An estimator."""

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return utils.pretty.format_object(self)

    def _set_params(self, **new_params):
        """Returns a new instance with the current parameters as well as new ones.

        The algorithm will be recursively called down ``Pipeline``s and ``TransformerUnion``s.

        Example:

            ::

                >>> from creme import linear_model
                >>> from creme import optim
                >>> from creme import preprocessing

                >>> model = (
                ...     preprocessing.StandardScaler() |
                ...     linear_model.LinearRegression(
                ...         optimizer=optim.SGD(lr=0.042),
                ...     )
                ... )

                >>> new_params = {
                ...     'LinearRegression': {
                ...         'l2': .001
                ...     }
                ... }

                >>> model._set_params(**new_params)
                Pipeline (
                  StandardScaler (
                    with_mean=True
                    with_std=True
                  ),
                  LinearRegression (
                    optimizer=SGD (
                      lr=Constant (
                        learning_rate=0.042
                      )
                    )
                    loss=Squared ()
                    l2=0.001
                    intercept=0.
                    intercept_lr=Constant (
                      learning_rate=0.01
                    )
                    clip_gradient=1e+12
                    initializer=Zeros ()
                  )
                )

        """

        # Get the input parameters, assuming that they are stored in the class
        params = {
            name: getattr(self, name)
            for name, param in inspect.signature(self.__class__).parameters.items()
            if param.kind != param.VAR_KEYWORD
        }

        # Add the new parameters
        params.update(new_params)

        # Return a new instance
        return self.__class__(**copy.deepcopy(params))

    @property
    def _tags(self) -> dict:
        """Returns the estimator's tags."""

        tags: dict = {}

        for base_class in inspect.getmro(self.__class__)[1:]:
            if isinstance(base_class, Estimator):
                _update_if_consistent(tags, base_class._more_tags)

        _update_if_consistent(tags, self._more_tags)

        return {**DEFAULT_TAGS, **tags}

    @property
    def _more_tags(self) -> dict:
        """Specific tags for this estimator."""
        return {}

    @property
    def _memory_usage(self) -> str:
        """Returns the memory usage in a human readable format."""

        def get_size(obj, seen=None):
            """Recursively finds size of objects"""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            # Important mark as seen *before* entering recursion to gracefully handle
            # self-referential objects
            seen.add(obj_id)
            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(vars(obj), seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            return size

        mem_usage = get_size(self)
        return utils.pretty.humanize_bytes(mem_usage)
