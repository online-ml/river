from collections.abc import Callable
from typing import TypeVar

import pytest

Benchmark = TypeVar("Benchmark", bound=Callable[..., object])


def benchmark(group: str) -> pytest.MarkDecorator:
    """Mark a test function as a CodSpeed benchmark in the given group."""
    return pytest.mark.benchmark(group=group)


def heavy(group: str) -> Callable[[Benchmark], Benchmark]:
    """Mark a CodSpeed benchmark as part of the heavy shard."""
    benchmark_mark = benchmark(group)

    def decorate(function: Benchmark) -> Benchmark:
        return pytest.mark.codspeed_heavy(benchmark_mark(function))

    return decorate
