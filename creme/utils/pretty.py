import inspect
import types

import numpy as np


__all__ = ['format_class', 'print_table']


def format_class(cls):
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
        rows = reversed(np.argsort(list(map(float, columns[headers.index(sort_by)]))))
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
