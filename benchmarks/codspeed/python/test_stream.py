import collections
import io

from river import stream

from marks import benchmark


NUMERIC_HEADER = """@relation test
@attribute a numeric
@attribute b numeric
@attribute c numeric
@attribute d numeric
@attribute e numeric
@attribute f numeric
@attribute g numeric
@attribute h numeric
@attribute i numeric
@attribute j numeric
@attribute label {negative,positive}
@data
"""
QUOTED_HEADER = NUMERIC_HEADER.replace("@attribute j numeric", "@attribute j string")
NUMERIC_DATA = (
    NUMERIC_HEADER + "0.213179,74,0,0.375607,3500,3,0,1,0,1,positive\n" * 20_000
)
QUOTED_DATA = QUOTED_HEADER + "1,2,3,4,5,6,7,8,9,'hello, world',positive\n" * 20_000


@benchmark("stream")
def test_iter_arff_dense_numeric(benchmark) -> None:
    benchmark(
        lambda: collections.deque(
            stream.iter_arff(io.StringIO(NUMERIC_DATA), target="label"), maxlen=0
        )
    )


@benchmark("stream")
def test_iter_arff_dense_quoted(benchmark) -> None:
    benchmark(
        lambda: collections.deque(
            stream.iter_arff(io.StringIO(QUOTED_DATA), target="label"), maxlen=0
        )
    )
