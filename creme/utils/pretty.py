"""Helper functions for making things readable by humans."""
import inspect
import types

import numpy as np


__all__ = ['format_object', 'print_table']


def format_object(obj, show_modules=False, depth=0):
    """Returns a pretty representation of an instanted object."""

    rep = f'{obj.__class__.__name__} ('
    if show_modules:
        rep = f'{obj.__class__.__module__}.{rep}'
    tab = '\t'

    init = inspect.signature(obj.__init__)
    n_params = 0

    for name, param in init.parameters.items():

        # We can't guess args and kwargs and so we don't handle them
        if (
            param.name == 'args' and param.kind == param.VAR_POSITIONAL or
            param.name == 'kwargs' and param.kind == param.VAR_KEYWORD
        ):
            continue

        # Retrieve the attribute associated with the parameter
        attr = getattr(obj, name)
        n_params += 1

        # Prettify the attribute when applicable
        if isinstance(attr, types.FunctionType):
            attr = attr.__name__
        if isinstance(attr, str):
            attr = f'"{attr}"'
        elif isinstance(attr, float):
            attr = (
                f'{attr:.0e}'
                if (attr > 1e5 or (attr < 1e-4 and attr > 0)) else
                f'{attr:.6f}'.rstrip('0')
            )
        elif isinstance(attr, set):
            attr = sorted(attr)
        elif hasattr(attr, '__class__') and 'creme.' in str(type(attr)):
            attr = format_object(obj=attr, show_modules=show_modules, depth=depth + 1)

        rep += f'\n{tab * (depth + 1)}{name}={attr}'

    if n_params:
        rep += f'\n{tab * depth}'
    rep += ')'

    return rep.expandtabs(2)


def print_table(headers, columns, sort_by=None):
    """Pretty-prints a table.

    Parameters:
        headers (list of str): The column names.
        columns (list of lists of str): The column values.
        sort_by (str): The name of the column by which to sort by.

    """

    # Check inputs
    if len(headers) != len(columns):
        raise ValueError('there must be as many headers as columns')

    if len(set(map(len, columns))) > 1:
        raise ValueError('all the columns must be of the same length')

    # Determine the width of each column based on the maximum length of it's elements
    col_widths = [
        max(*map(len, col), len(header))
        for header, col in zip(headers, columns)
    ]

    # Make a template to print out rows one by one
    row_format = ' '.join(['{:' + str(width + 2) + 's}' for width in col_widths])

    # Determine in which order to print the rows
    if sort_by is not None:
        rows = reversed(np.argsort(list(columns[headers.index(sort_by)])))
    else:
        rows = range(len(columns[0]))

    # Build the table
    table = (
        row_format.format(*headers) + '\n' +
        '\n'.join((
            row_format.format(*[
                col[i].rjust(width)
                for col, width in zip(columns, col_widths)
            ])
            for i in rows
        ))
    )

    return table
