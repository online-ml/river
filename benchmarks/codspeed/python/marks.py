import pytest


def benchmark(group: str) -> pytest.MarkDecorator:
    """Mark a test function as a CodSpeed benchmark in the given group."""
    return pytest.mark.benchmark(group=group)
