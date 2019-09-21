import inspect
import types


__all__ = ['pretty_format_class']


def pretty_format_class(cls):
    """Returns a pretty representation of a class."""

    rep = f'{cls.__class__.__name__} ('
    init = inspect.signature(cls.__init__)

    for name, param in init.parameters.items():

        # Retrieve the attribute associated with the parameter
        if param.default is None or param.default == param.empty:
            try:
                attr = getattr(cls, name)
            except AttributeError:
                continue
        else:
            attr = param.default

        # Prettify the attribute when applicable
        if isinstance(attr, str):
            attr = f"'{attr}'"
        elif isinstance(attr, types.FunctionType):
            attr = attr.__name__
        elif isinstance(attr, set):
            attr = sorted(attr)

        rep += f'\n    {name}={attr}'

    if init.parameters:
        rep += '\n)'
    else:
        rep += ')'

    return rep
