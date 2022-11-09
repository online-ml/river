"""Helper functions for making things readable by humans."""

from __future__ import annotations

import math

__all__ = ["humanize_bytes", "print_table"]


def print_table(
    headers: list[str],
    columns: list[list[str]],
    order: list[int] = None,
):
    """Pretty-prints a table.

    Parameters
    ----------
    headers
        The column names.
    columns
        The column values.
    order
        Order in which to print the column the values. Defaults to the order in which the values
        are given.

    """

    # Check inputs
    if len(headers) != len(columns):
        raise ValueError("there must be as many headers as columns")

    if len(set(map(len, columns))) > 1:
        raise ValueError("all the columns must be of the same length")

    # Determine the width of each column based on the maximum length of it's elements
    col_widths = [max(*map(len, col), len(header)) for header, col in zip(headers, columns)]

    # Make a template to print out rows one by one
    row_format = " ".join(["{:" + str(width + 2) + "s}" for width in col_widths])

    # Determine the order in which to print the column values
    if order is None:
        order = list(range(len(columns[0])))

    # Build the table
    table = (
        row_format.format(*headers)
        + "\n"
        + "\n".join(
            row_format.format(*[col[i].rjust(width) for col, width in zip(columns, col_widths)])
            for i in order
        )
    )

    return table


def humanize_bytes(n_bytes: int):
    """Returns a human-friendly byte size.

    Parameters
    ----------
    n_bytes

    """
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    human = float(n_bytes)
    rank = 0
    if n_bytes != 0:
        rank = int((math.log10(n_bytes)) / 3)
        rank = min(rank, len(suffixes) - 1)
        human = n_bytes / (1024.0**rank)
    f = ("%.2f" % human).rstrip("0").rstrip(".")
    return f"{f} {suffixes[rank]}"
